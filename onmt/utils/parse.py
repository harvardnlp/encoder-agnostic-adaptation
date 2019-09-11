import configargparse as cfargparse
import os

import torch

import onmt.opts as opts
from onmt.utils.logging import logger


class ArgumentParser(cfargparse.ArgumentParser):
    def __init__(
            self,
            config_file_parser_class=cfargparse.YAMLConfigFileParser,
            formatter_class=cfargparse.ArgumentDefaultsHelpFormatter,
            **kwargs):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs)

    @classmethod
    def defaults(cls, *args):
        """Get default arguments added to a parser by all ``*args``."""
        dummy_parser = cls()
        for callback in args:
            callback(dummy_parser)
        defaults = dummy_parser.parse_known_args([])[0]
        return defaults

    @classmethod
    def update_model_opts(cls, model_opt):
        if model_opt.word_vec_size > 0:
            model_opt.src_word_vec_size = model_opt.word_vec_size
            model_opt.tgt_word_vec_size = model_opt.word_vec_size

        if model_opt.layers > 0:
            model_opt.enc_layers = model_opt.layers
            model_opt.dec_layers = model_opt.layers

        if model_opt.rnn_size > 0:
            model_opt.enc_rnn_size = model_opt.rnn_size
            model_opt.dec_rnn_size = model_opt.rnn_size

        if model_opt.heads > 0:
            model_opt.enc_heads = model_opt.heads
            model_opt.dec_heads = model_opt.heads

        model_opt.brnn = model_opt.encoder_type == "brnn"

        if model_opt.copy_attn_type is None:
            model_opt.copy_attn_type = model_opt.global_attention

        if model_opt.position_encoding_learned:
            model_opt.position_encoding_learned_enc = True
            model_opt.position_encoding_learned_dec = True
        
    @classmethod
    def validate_model_opts(cls, model_opt):
        assert model_opt.model_type in ["text", "img", "audio", 'imgvec', 'none'], \
            "Unsupported model type %s" % model_opt.model_type

        # this check is here because audio allows the encoder and decoder to
        # be different sizes, but other model types do not yet
        same_size = model_opt.enc_rnn_size == model_opt.dec_rnn_size
        assert model_opt.model_type == 'audio' or same_size, \
            "The encoder and decoder rnns must be the same size for now"

        assert model_opt.rnn_type != "SRU" or model_opt.gpu_ranks, \
            "Using SRU requires -gpu_ranks set."
        if model_opt.share_embeddings:
            if model_opt.model_type != "text":
                raise AssertionError(
                    "--share_embeddings requires --model_type text.")
        if model_opt.model_dtype == "fp16":
            logger.warning(
                "FP16 is experimental, the generated checkpoints may "
                "be incompatible with a future version")

        if model_opt.share_position_embeddings and not model_opt.position_encoding_learned:
            raise AssertionError('It does not make sense to share position embeddings if '
                                 'they are not learned')
        if int(model_opt.use_GPT_version_psa) + int(model_opt.use_GPT_version_unconditional) + \
           int(model_opt.use_GPT_version_ctxattn)> 1:
            raise AssertionError('At most one of use_GPT_version, use_GPT_version_alt, '
                                 'use_GPT_version_psa, use_GPT_version_unconditional, '
                                 'use_GPT_version_ctxattn can be true at the same time')

        if model_opt.simple_fusion and model_opt.gpt2_params_path is None:
            raise AssertionError('Simple fusion requires setting the gpt2_params_path option')
        
        if model_opt.attn_hidden > 0:
            raise NotImplementedError

        if model_opt.GPT_representation_mode != 'none' and (model_opt.gpt2_init_embanddec or model_opt.simple_fusion or model_opt.gpt2_init_embandenc):
            raise AssertionError('loading GPT weights for seq2seq initialization AND GPT '
                                 'probably does not make sense')

    @classmethod
    def ckpt_model_opts(cls, ckpt_opt):
        # Load default opt values, then overwrite with the opts in
        # the checkpoint. That way, if there are new options added,
        # the defaults are used.
        opt = cls.defaults(opts.model_opts)
        opt.__dict__.update(ckpt_opt.__dict__)
        return opt

    @classmethod
    def validate_train_opts(cls, opt):
        if opt.epochs:
            raise AssertionError(
                "-epochs is deprecated please use -train_steps.")
        if opt.truncated_decoder > 0 and opt.accum_count > 1:
            raise AssertionError("BPTT is not compatible with -accum > 1")
        if opt.gpuid:
            raise AssertionError("gpuid is deprecated \
                  see world_size and gpu_ranks")
        if torch.cuda.is_available() and not opt.gpu_ranks:
            logger.info("WARNING: You have a CUDA device, \
                        should run with -gpu_ranks")

        if opt.gpt2_params_path is not None and \
            not (opt.gpt2_init_embanddec or opt.gpt2_init_embandenc) and opt.gpt2_params_std <= 0 \
            and not opt.simple_fusion and opt.GPT_representation_mode == 'none':
            raise AssertionError('Loading GPT parameters but not doing anything with them!')

        if (opt.gpt2_init_embanddec or opt.gpt2_params_std > 0) and opt.gpt2_params_path is None:
            raise AssertionError('Trying to use gpt2 parameters, but gpt2_params_path is not given')

        if opt.train_from and opt.gpt2_init_embanddec:
            raise AssertionError('Trying to initialize gpt2 weights while also loading a save file. This is likely a mistake.')

        if opt.train_from and opt.encoder_from:
            raise AssertionError('Trying to initialize encoder weights while also loading a save file. This is likely a mistake.')

        if opt.attn_dropout < 0:
            opt.attn_dropout = opt.dropout

        if opt.train_from and opt.load_uncond_from:
            raise AssertionError('Only one of train_from, load_uncond_from makes sense')


    @classmethod
    def validate_translate_opts(cls, opt):
        if opt.beam_size != 1 and opt.random_sampling_topk != 1:
            raise ValueError('Can either do beam search OR random sampling.')

    @classmethod
    def validate_preprocess_args(cls, opt):
        assert opt.max_shard_size == 0, \
            "-max_shard_size is deprecated. Please use \
            -shard_size (number of examples) instead."
        assert opt.shuffle == 0, \
            "-shuffle is not implemented. Please shuffle \
            your data before pre-processing."

        assert (opt.data_type == 'none' or os.path.isfile(opt.train_src)) \
            and os.path.isfile(opt.train_tgt), \
            "Please check path of your train src and tgt files!"

        assert not opt.valid_src or os.path.isfile(opt.valid_src), \
            "Please check path of your valid src file!"
        assert not opt.valid_tgt or os.path.isfile(opt.valid_tgt), \
            "Please check path of your valid tgt file!"

        if opt.free_src and not opt.fixed_vocab:
            raise ValueError('free_src only makes sense when using fixed_vocab')

        assert not (opt.data_type == 'imgvec' and opt.shard_size > 0)
