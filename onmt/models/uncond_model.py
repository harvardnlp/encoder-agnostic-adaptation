""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch

class UncondModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, decoder):
        super(UncondModel, self).__init__()
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, **kwargs):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        memory_bank = torch.zeros((1, tgt.shape[1], 1), dtype=torch.float, device=tgt.device)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, None)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths, **kwargs)
        return dec_out, attns
