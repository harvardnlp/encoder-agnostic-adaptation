#!/usr/bin/env python
"""Training on a single process."""
import os

import torch

from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model, only_trainable=False):
    enc = 0
    dec = 0
    lm_dec = 0
    for name, param in model.named_parameters():
        if only_trainable and not param.requires_grad:
            continue
        if 'lm_decoder' in name:
            lm_dec += param.nelement()
        elif 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec + lm_dec, enc, dec, lm_dec


def configure_process(opt, device_id):
    if device_id >= 0:
        try:
            torch.cuda.set_device(device_id)
        except AttributeError:
            print("Failed to set CUDA device, using CPU")
    set_random_seed(opt.seed, device_id >= 0)


def main(opt, device_id):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    load_str = opt.train_from if opt.train_from else opt.load_uncond_from
    if load_str:
        logger.info('Loading checkpoint from %s' % load_str)
        checkpoint = torch.load(load_str,
                                map_location=lambda storage, loc: storage)

        logger.info('Loading vocab from checkpoint at %s.' % load_str)
        vocab = checkpoint['vocab']

        if opt.train_from:
            model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
            ArgumentParser.update_model_opts(model_opt)
            ArgumentParser.validate_model_opts(model_opt)
        else:
            model_opt = opt
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + '.vocab.pt')

    if opt.gpt2_params_path is not None:
        import tensorflow as tf
        import numpy as np
        # Taken from pytorch-pretrained-BERT:
        # Load weights from TF model
        logger.info("Loading TF GPT weights...")
        init_vars = tf.train.list_variables(opt.gpt2_params_path)
        names = []
        arrays = []
        for name, shape in init_vars:
            if opt.gpt_emb_only and ('wpe' not in name and 'wte' not in name):
                continue
            if opt.gpt_wpe_only and 'wpe' not in name:
                continue
            #print("Loading TF weight {} with shape {}".format(name, shape))
            array = tf.train.load_variable(opt.gpt2_params_path, name)
            names.append(name)
            arrays.append(array.squeeze())
        logger.info("Done.")
        
        if checkpoint is None:
            checkpoint = {'gpt2_params': zip(names, arrays)}
        else:
            checkpoint['gpt2_params'] = zip(names, arrays)

    if opt.encoder_from is not None:
        logger.info('Loading checkpoint with encoder from %s' % opt.encoder_from)
        enc_checkpoint = torch.load(opt.encoder_from,
                                map_location=lambda storage, loc: storage)
        enc_vocab = enc_checkpoint['vocab']
        if vocab['src'].base_field.vocab != enc_vocab['src'].base_field.vocab:
            raise ValueError('encoder vocab and model vocab need to be identical it using pretrained encoder')
        if checkpoint is None:
            checkpoint = {}
        checkpoint['enc_model'] = enc_checkpoint['model']
        
    
    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # Report src and tgt vocab sizes, including for features
    sides = ['tgt'] if opt.model_type == 'none' else ['src', 'tgt']
    for side in sides:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec, lm_dec = _tally_parameters(model)
    n_params_t, enc_t, dec_t, lm_dec_t = _tally_parameters(model, only_trainable=True)
    logger.info('encoder: %d (%d)' % (enc, enc_t))
    logger.info('decoder: %d (%d)' % (dec, dec_t))
    if opt.simple_fusion:
        logger.info('lm decoder: %d (%d)' % (lm_dec, lm_dec_t))

    logger.info('* number of parameters: %d (%d)' % (n_params, n_params_t))
    _check_save_model_path(opt)

    if not opt.train_from and opt.gpt2_params_path is not None:
        checkpoint = None

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, device_id, model, fields, optim, model_saver=model_saver)

    train_iter = build_dataset_iter("train", fields, opt)
    valid_iter = build_dataset_iter(
        "valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0
    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()
