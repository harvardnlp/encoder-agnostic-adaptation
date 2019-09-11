# -*- coding: utf-8 -*-

import os
import numpy as np

import torch
from torchtext.data import Field

from onmt.inputters.datareader_base import DataReaderBase


class NoneDataReader(DataReaderBase):
    """Read image data vectors from disk.

    Args:
        channel_size (int): Number of channels per image.
    """
    def read(self, path, side, img_dir=None):
        """Read data into dicts.

        Args:
            path (str): Path to npy file with saved image vectors
                The filenames may be relative to ``src_dir``
                (default behavior) or absolute.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.

        Yields:
            a dictionary containing image data and index for each line.
        """

        for i in range(features.shape[0]):
            yield {side: torch.tensor(features[i]), 'indices': i}

def img_vec_sort_key(ex):
    """Sort using the number of image box features."""
    return ex.src.size(0)

def batch_img_vec(data, vocab):
    """Batch a sequence of image vectors."""
    imgs = torch.stack(data, dim=1) # [K, B, dim]
    return imgs

def image_vec_fields(**kwargs):
    img = Field(
        use_vocab=False, dtype=torch.float,
        postprocessing=batch_img_vec, sequential=False)
    return img
