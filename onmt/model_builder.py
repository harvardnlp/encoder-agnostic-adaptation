"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import math
import copy
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders import str2enc

from onmt.decoders import str2dec

from onmt.modules import Embeddings, CopyGenerator, SimpleFusionGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser


def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]

    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    fix_word_vecs = opt.fix_word_vecs_enc if for_encoder \
        else opt.fix_word_vecs_dec

    pos_enc_learned = opt.position_encoding_learned_enc if for_encoder else opt.position_encoding_learned_dec
    GPT_representation_mode = opt.GPT_representation_mode if opt.GPT_representation_loc == 'both' or (opt.GPT_representation_loc == 'src' and for_encoder) or (opt.GPT_representation_loc == 'tgt' and not for_encoder) else 'none'

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        position_encoding_learned=pos_enc_learned,
        position_encoding_ctxsize=opt.position_encoding_ctxsize,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        fix_word_vecs=fix_word_vecs,
        GPT_representation_mode=GPT_representation_mode,
        GPT_representation_tgt=not for_encoder
    )
    return emb


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    enc_type = opt.encoder_type if opt.model_type == "text" else opt.model_type
    return str2enc[enc_type].from_opt(opt, embeddings)


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed \
               else opt.decoder_type
    return str2dec[dec_type].from_opt(opt, embeddings)


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(
            vocab, opt.data_type, dynamic_dict=model_opt.copy_attn
        )
    else:
        fields = vocab

    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint,
                             opt.gpu)
    if opt.fp32:
        model.float()

    model.eval()
    model.generator.eval()
    return fields, model, model_opt

class PadGen(nn.Module):
    def __init__(self):
        super(PadGen, self).__init__()

    def forward(self, vals):
        # Need to make this more general
        vals[..., 50257:] = -1e10
        return vals
        

def build_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # Build embeddings.
    if model_opt.model_type == "text":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None

    # Build encoder.
    encoder = build_encoder(model_opt, src_emb)

    # Build decoder.
    tgt_field = fields["tgt"]
    tgt_emb = build_embeddings(model_opt, tgt_field, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        assert src_field.base_field.vocab == tgt_field.base_field.vocab, \
            "preprocess with -share_vocab if you use share_embeddings"

        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    if model_opt.share_position_embeddings:
        tgt_emb.make_embedding.pe.pe.weight = src_emb.make_embedding.pe.pe.weight

    decoder = build_decoder(model_opt, tgt_emb)

    # Build NMTModel(= encoder + decoder).
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")

    # Build separate LM if doing simple fusion
    if model_opt.simple_fusion:
        layers = 12
        size = 768
        heads = 12

        lm_decoder_opt = copy.deepcopy(model_opt)
        lm_decoder_opt.dec_layers = layers
        lm_decoder_opt.use_GPT_version_ctxattn = False
        lm_decoder_opt.use_GPT_version_psa = False
        lm_decoder_opt.use_GPT_version_unconditional = True
        lm_decoder_opt.tgt_word_vec_size = size
        lm_decoder_opt.rnn_size = size
        lm_decoder_opt.dec_rnn_size = size
        lm_decoder_opt.transformer_ff = size*4
        lm_decoder_opt.dec_heads = heads
        lm_decoder_opt.position_encoding_learned_dec = True
        lm_decoder_opt.share_decoder_embeddings = True
        lm_decoder_opt.dropout = 0

        lm_decoder_emb = build_embeddings(lm_decoder_opt, tgt_field, for_encoder=False)
        logger.info(lm_decoder_emb)

        lm_decoder = build_decoder(lm_decoder_opt, lm_decoder_emb)
        load_decoder = lm_decoder

        model = onmt.models.SimpleFusionModel(encoder, decoder, lm_decoder)

        generator = SimpleFusionGenerator(model_opt.dec_rnn_size,
                                          lm_decoder_opt.dec_rnn_size,
                                          len(fields["tgt"].base_field.vocab))
        generator.lm_linear.weight = lm_decoder.embeddings.word_lut.weight

        if model_opt.share_decoder_embeddings:
            generator.decoder_linear.weight = decoder.embeddings.word_lut.weight
        gen_linear = generator.lm_linear
    else:
        load_decoder = decoder
        if model_opt.unconditional:
            model = onmt.models.UncondModel(decoder)
        else:
            model = onmt.models.NMTModel(encoder, decoder)

        # Build Generator.
        if not model_opt.copy_attn:
            if model_opt.generator_function == "sparsemax":
                gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
            else:
                gen_func = nn.LogSoftmax(dim=-1)

            if model_opt.padded_vocab_fix_me_later:
                gen_func = nn.Sequential(PadGen(), gen_func)

            generator = nn.Sequential(
                nn.Linear(model_opt.dec_rnn_size,
                          len(fields["tgt"].base_field.vocab)),
                Cast(torch.float32),
                gen_func
            )
            if model_opt.share_decoder_embeddings:
                generator[0].weight = decoder.embeddings.word_lut.weight
            gen_linear = generator[0]
        else:
            tgt_base_field = fields["tgt"].base_field
            vocab_size = len(tgt_base_field.vocab)
            pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
            generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)
            if model_opt.share_decoder_embeddings:
                generator.linear.weight = decoder.embeddings.word_lut.weight
            gen_linear = generator.linear

    if model_opt.encdec_share_params:
        for name, p in decoder.named_parameters():
            if 'ctx' in name or 'context' in name:
                continue
            pointer = encoder
            attrs = name.split('.')
            for attr_name in attrs[:-1]:
                pointer = getattr(pointer, attr_name)

            # pointer now has the encoder version of the parameter parent
            setattr(pointer, attrs[-1], p)


    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # Normally, just load the model parameters from checkpoint
        if 'gpt2_params' not in checkpoint and 'enc_model' not in checkpoint:
            # This preserves backward-compat for models using customed layernorm
            def fix_key(s):
                s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                           r'\1.layer_norm\2.bias', s)
                s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                           r'\1.layer_norm\2.weight', s)
                return s
            
            checkpoint['model'] = {fix_key(k): v
                                   for k, v in checkpoint['model'].items()}
            # end of patch for backward compatibility

            # Initialize rest of parameters normally
            if hasattr(model_opt, 'load_uncond_from') and model_opt.load_uncond_from:
                for p in decoder.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                
                # Always initialize encoder parameters normally
                for p in encoder.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

                if model_opt.ctx_weight_param:
                    for name, p in decoder.named_parameters():
                        if 'ctx_weight' in name:
                            p.data.zero_()
                        if 'ctx_bias' in name:
                            p.data.fill_(-10)


            model.load_state_dict(checkpoint['model'], strict=False)
            generator.load_state_dict(checkpoint['generator'], strict=False)
        else:
            # load the gpt parameters
            if 'gpt2_params' in checkpoint:
                init_something = model_opt.gpt2_init_embanddec or model_opt.simple_fusion or model_opt.gpt2_init_embandenc or model_opt.GPT_representation_mode != 'none'
                
                if init_something:
                    # Initialize all the weights first
                    if model_opt.gpt2_init_zero:
                        for p in decoder.parameters():
                            p.data.zero_()
                        if model_opt.simple_fusion:
                            generator.decoder_linear.weight.data.zero_()
                            generator.decoder_linear.bias.data.zero_()
                    else:
                        for p in decoder.parameters():
                            if p.dim() > 1:
                                xavier_uniform_(p)
                    
                    # Always initialize encoder parameters normally
                    if encoder is not None:
                        for p in encoder.parameters():
                            if p.dim() > 1:
                                xavier_uniform_(p)
                    for p in generator.parameters():
                        if p.dim() > 1:
                            xavier_uniform_(p)
                    if model_opt.zero_bias_init:
                        gen_linear.bias.data.zero_()

                    if model_opt.ctx_weight_param:
                        for name, p in decoder.named_parameters():
                            if 'ctx_weight' in name:
                                p.data.zero_()
                            if 'ctx_bias' in name:
                                p.data.fill_(-10)
                        gen_linear.bias.data.zero_()

                load_models = []
                if model_opt.GPT_representation_mode != 'none':
                    load_embs = []
                    if model_opt.GPT_representation_loc in ['both', 'src']:
                        load_models.append(src_emb.gpt_model)
                        load_embs.append(src_emb)
                    if model_opt.GPT_representation_loc in ['both', 'tgt']:
                        load_models.append(tgt_emb.gpt_model)
                        load_embs.append(tgt_emb)
                    
                else:
                    if model_opt.gpt2_init_embanddec or model_opt.simple_fusion:
                        load_models = [load_decoder]
                    elif model_opt.gpt2_init_embandenc:
                        load_models = [encoder]
                
                it_list = list(checkpoint['gpt2_params'])
                for lm_idx, load_model in enumerate(load_models):
                    #print(lm_idx, load_model)
                    for name, array in it_list:
                        name = name[6:]  # skip "model/"
                        name = name.split('/')

                        assigned = False
                        if name[0] == 'wpe':
                            if model_opt.GPT_representation_mode != 'none':
                                pointer = load_embs[lm_idx].make_embedding.pe.pe.weight
                            else:
                                pointer = load_model.embeddings.make_embedding.pe.pe.weight

                        elif name[0] == 'wte':
                            if model_opt.GPT_representation_mode != 'none':
                                pointer = [load_embs[lm_idx].make_embedding.emb_luts[0].weight, gen_linear.weight]
                            else:
                                pointer = [load_model.embeddings.make_embedding.emb_luts[0].weight]
                                if not model_opt.nopretrain_decemb:
                                    pointer.append(gen_linear.weight)
                                if model_opt.simple_fusion and model_opt.sf_pretrain_dec_emb:
                                    pointer.append(decoder.embeddings.make_embedding.emb_luts[0].weight)

                        elif name[0] == 'ln_f':
                            if name[1] == 'g':
                                pointer = load_model.layer_norm.weight
                            elif name[1] == 'b':
                                pointer = load_model.layer_norm.bias
                            else:
                                raise ValueError('I am missing something here!')

                        elif name[0][0] == 'h':
                            layer_num = name[0][1:]
                            pointer = getattr(load_model.transformer_layers, layer_num)
                            if name[1] == 'attn':
                                assigned = True
                                pointer = pointer.self_attn
                                full_data = torch.from_numpy(array)
                                if name[2] == 'c_attn':
                                    end_size = full_data.shape[-1]//3
                                    assert full_data.shape[-1] % 3 == 0
                                    if name[3] == 'b':
                                        if init_something:
                                            pointer.linear_query.bias.data = full_data[:end_size]
                                            pointer.linear_keys.bias.data = full_data[end_size:end_size*2]
                                            pointer.linear_values.bias.data = full_data[end_size*2:]
                                        if model_opt.gpt2_params_std > 0:
                                            pointer.linear_query.bias.orig = full_data[:end_size].clone()
                                            pointer.linear_keys.bias.orig = full_data[end_size:end_size*2].clone()
                                            pointer.linear_values.bias.orig = full_data[end_size*2:].clone()
                                    elif name[3] == 'w':
                                        if init_something:
                                            pointer.linear_query.weight.data = full_data[:, :end_size].t().contiguous()
                                            pointer.linear_keys.weight.data = full_data[:, end_size:end_size*2].t().contiguous()
                                            pointer.linear_values.weight.data = full_data[:, end_size*2:].t().contiguous()
                                        if model_opt.gpt2_params_std > 0:
                                            pointer.linear_query.weight.orig = full_data[:, :end_size].t().contiguous().clone()
                                            pointer.linear_keys.weight.orig = full_data[:, end_size:end_size*2].t().contiguous().clone()
                                            pointer.linear_values.weight.orig = full_data[:, end_size*2:].t().contiguous().clone()
                                    else:
                                        raise ValueError('I am missing something here!')
                                elif name[2] == 'c_proj':
                                    if name[3] == 'b':
                                        if init_something:
                                            pointer.final_linear.bias.data = full_data
                                        if model_opt.gpt2_params_std > 0:
                                            pointer.final_linear.bias.orig = full_data.clone()
                                    elif name[3] == 'w':
                                        if init_something:
                                            pointer.final_linear.weight.data = full_data.t().contiguous()
                                        if model_opt.gpt2_params_std > 0:
                                            pointer.final_linear.weight.orig = full_data.t().contiguous().clone()

                                    else:
                                        raise ValueError('I am missing something here!')

                            elif name[1] == 'ln_1' or name[1] == 'ln_2':
                                num = name[1][3]
                                pointer = getattr(pointer, 'layer_norm_'+num)
                                if name[2] == 'b':
                                    pointer = pointer.bias
                                elif name[2] == 'g':
                                    pointer = pointer.weight
                                else:
                                    raise ValueError('I am missing something here!')
                            elif name[1] == 'mlp':
                                pointer = pointer.feed_forward
                                pointer = getattr(pointer, name[2])
                                if name[3] == 'b':
                                    pointer = pointer.bias
                                elif name[3] == 'w':
                                    pointer = pointer.weight
                                else:
                                    raise ValueError('I am missing something here!')
                            else:
                                raise ValueError('I am missing something here!')
                        else:
                            raise ValueError('I am missing something here!')
                        
                        if not assigned:
                            if name[-1] == 'w' or name[-1] == 'g':
                                array = array.T

                            if not isinstance(pointer, list):
                                pointer = [pointer]
                            for pointer_i in pointer:
                                target_size = int(math.ceil(array.shape[0]/8))*8
                                padded_vocab = name[0] == 'wte' and pointer_i.shape[0] == target_size
                                padded_vocab = padded_vocab and pointer_i.shape[1:] == array.shape[1:]
                                try:
                                    assert pointer_i.shape == array.shape or padded_vocab
                                except AssertionError as e:
                                    e.args += (pointer_i.shape, array.shape)
                                    raise
                                if init_something:
                                    print("Initialize PyTorch weight {}".format(name))
                                    if padded_vocab:
                                        pointer_i.data[:array.shape[0]] = torch.from_numpy(array)
                                    else:
                                        pointer_i.data = torch.from_numpy(array)
                                if model_opt.gpt2_params_std > 0:
                                    if padded_vocab:
                                        raise NotImplementedError
                                    else:
                                        pointer_i.orig = torch.from_numpy(array).clone()
            if 'enc_model' in checkpoint:
                load_dict = {k[8:]: v for k, v in checkpoint['enc_model'] if 'encoder' in k}
                encoder.load_state_dict(load_dict, strict=True)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if not model_opt.unconditional and hasattr(model.encoder, 'embeddings') \
                and model.encoder.embeddings is not None:
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec)

    # remove requires_grad from params that are not trained:
    if model_opt.notrain_emb or model_opt.notrain_embanddec:
        if model_opt.position_encoding_learned_enc and model_opt.share_position_embeddings:
            model.encoder.embeddings.make_embedding.pe.pe.weight.requires_grad = False
        if model_opt.share_embeddings:
            model.encoder.embeddings.make_embedding.emb_luts[0].weight.requires_grad = False
        model.decoder.embeddings.make_embedding.pe.pe.weight.requires_grad = False
        model.decoder.embeddings.make_embedding.emb_luts[0].weight.requires_grad = False
        generator[0].weight.requires_grad = False

    if model_opt.notrain_genbias:
        generator[0].bias.requires_grad = False

    if model_opt.notrain_embanddec:
        for name, p in load_decoder.layer_norm.named_parameters():
            p.requires_grad = False
        for name, p in load_decoder.transformer_layers.named_parameters():
            if 'context' not in name and 'ctx' not in name: # Takes care of normal and psa versions
                p.requires_grad = False
    
    if model_opt.onlytrainln:
        for name, p in model.decoder.named_parameters():
            if 'layer_norm' not in name:
                p.requires_grad = False
        for p in generator.parameters():
            p.requires_grad = False

    if model_opt.onlytrainoutp:
        if model_opt.share_decoder_embeddings:
            raise ValueError

        for p in model.decoder.parameters():
            p.requires_grad = False

    if model_opt.simple_fusion:
        for p in lm_decoder.parameters():
            p.requires_grad = False
        for p in generator.lm_linear.parameters():
            p.requires_grad = False

    model.generator = generator
    model.to(device)
    if model_opt.model_dtype == 'fp16':
        model.half()

    for p in model.parameters():
        if hasattr(p, 'orig'):
            p.orig = p.orig.to(device)
            if model_opt.model_dtype == 'fp16':
                p.orig = p.orig.half()

    return model


def linear_repr_patch(self):
    return 'in_features={}, out_features={}, bias={}, wgrad={}, bgrad={}'.format(
        self.in_features, self.out_features, self.bias is not None,
        self.weight.requires_grad, self.bias.requires_grad if self.bias is not None else 'N/A'
    )

def ln_repr_patch(self):
    string = '{normalized_shape}, eps={eps}, ' \
        'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
    string += ', wgrad={}, bgrad={}'.format(self.weight.requires_grad if self.weight is not None else 'N/A', 
            self.bias.requires_grad if self.bias is not None else 'N/A')
    return string

def emb_repr_patch(self):
    s = '{num_embeddings}, {embedding_dim}'
    if self.padding_idx is not None:
        s += ', padding_idx={padding_idx}'
    if self.max_norm is not None:
        s += ', max_norm={max_norm}'
    if self.norm_type != 2:
        s += ', norm_type={norm_type}'
    if self.scale_grad_by_freq is not False:
        s += ', scale_grad_by_freq={scale_grad_by_freq}'
    if self.sparse is not False:
        s += ', sparse=True'
    s = s.format(**self.__dict__)
    s += ', grad={}'.format(self.weight.requires_grad)
    return s

def build_model(model_opt, opt, fields, checkpoint):
    logger.info('Building model...')
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)

    # Show which params will be updated
    nn.Linear.extra_repr = linear_repr_patch
    nn.LayerNorm.extra_repr = ln_repr_patch
    nn.Embedding.extra_repr = emb_repr_patch

    logger.info(model)
    return model
