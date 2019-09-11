"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from onmt.inputters.inputter import \
    load_old_vocab, get_fields, OrderedIterator, \
    build_vocab, old_style_vocab, filter_example
from onmt.inputters.dataset_base import Dataset
from onmt.inputters.text_dataset import text_sort_key, TextDataReader
from onmt.inputters.image_dataset import img_sort_key, ImageDataReader
from onmt.inputters.audio_dataset import audio_sort_key, AudioDataReader
from onmt.inputters.image_vec_dataset import img_vec_sort_key, ImageVecDataReader
from onmt.inputters.datareader_base import DataReaderBase


str2reader = {
    "text": TextDataReader, "img": ImageDataReader,
    "audio": AudioDataReader, 'imgvec': ImageVecDataReader}

def none_sort(ex):
    return len(ex.tgt[0]) if hasattr(ex, 'tgt') else 0
str2sortkey = {
    'text': text_sort_key, 'img': img_sort_key,
    'audio': audio_sort_key, 'imgvec': img_vec_sort_key,
    'none': none_sort}


__all__ = ['Dataset', 'load_old_vocab', 'get_fields', 'DataReaderBase',
           'filter_example', 'old_style_vocab',
           'build_vocab', 'OrderedIterator',
           'text_sort_key', 'img_sort_key', 'audio_sort_key',
           'TextDataReader', 'ImageDataReader', 'AudioDataReader',
           'img_vec_sort_key', 'ImageVecDataReader']
