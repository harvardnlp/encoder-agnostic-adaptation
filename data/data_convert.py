# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example of converting model data.
Usage:
python data_convert_example.py --command binary_to_text --in_file /path/to_bin/data.bin --out_file name_text_data.txt
python data_convert_example.py --command text_to_binary --in_file /path/to_text/text_data.txt --out_file data/name_binary_data(.bin)
"""

"""
Uses Tensorflow 1.x 
Modified version: 
I adapted the _binary_to_text function so that it outputs the format expected by NeuSum (https://github.com/magic282/NeuSum)
  article  => out_file.txt.src.txt
  abstract => out_file.txt.tgt.txt
  Rename the file (Necessary so that the encode_text.py can find data for BPEising it)
  article => out_file.txt.src
  abstract => out_file.txt.tgt  
"""


import struct
import sys
import tensorflow as tf
from tensorflow.core.example import example_pb2
import os
from nltk.tokenize import sent_tokenize, word_tokenize
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'binary_to_text',
                           'Either binary_to_text or text_to_binary.'
                           'Specify FLAGS.in_file accordingly.')
tf.app.flags.DEFINE_string('in_file', '', 'path to file')
tf.app.flags.DEFINE_string('out_file', '', 'path to file')


def _binary_to_text_for_neusum():
    reader = open(FLAGS.in_file, 'rb')
    writer_src = open("%s.src.txt" % FLAGS.out_file, 'w')
    writer_tgt = open("%s.tgt.txt" % FLAGS.out_file, 'w')
    while True:
        len_bytes = reader.read(8)
        if not len_bytes:
            sys.stderr.write('Done reading\n')
            return
        str_len = struct.unpack('q', len_bytes)[0]
        tf_example_str = struct.unpack(
            '%ds' % str_len, reader.read(str_len))[0]
        tf_example = example_pb2.Example.FromString(tf_example_str)
        src_sentences = sent_tokenize(
            "%s" % tf_example.features.feature["article"].bytes_list.value[0])

        # in this case we get rid off <s></s>
        #<s> harry potter star daniel radcliffe gets # 20m fortune as he turns 18 monday . </s> <s> young actor says he has no plans to fritter his cash away . </s> <s> radcliffe 's earnings from first five potter films have been held in trust fund . </s>
        tgt_txt =  "%s" % tf_example.features.feature["abstract"].bytes_list.value[0]
        tgt_txt =  tgt_txt.replace(" </s> <s> ", "##SENT##")
        tgt_txt =  tgt_txt.replace("<s> ", "")
        tgt_txt =  tgt_txt.replace(" </s>", "")

        writer_src.write("##SENT##".join(src_sentences) + os.linesep)
        writer_tgt.write(tgt_txt + os.linesep)


        #examples = []
        # for key in tf_example.features.feature:
        #    examples.append('%s=%s' % (
        #        key, tf_example.features.feature[key].bytes_list.value[0]))
        # writer.write('%s\n' % '\t'.join(examples))
    reader.close()
    writer_src.close()
    writer_tgt.close()


def _text_to_binary():
    inputs = open(FLAGS.in_file, 'r').readlines()
    writer = open(FLAGS.out_file, 'wb')
    for inp in inputs:
        tf_example = example_pb2.Example()
        for feature in inp.strip().split('\t'):
            (k, v) = feature.split('=')
            if k.startswith('"') and k.endswith('"'):
                k = k[1:-1]
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            tf_example.features.feature[k.encode(
                'utf8')].bytes_list.value.extend([v.encode('utf8')])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))
    writer.close()


def main(unused_argv):
    assert FLAGS.command and FLAGS.in_file and FLAGS.out_file
    if FLAGS.command == 'binary_to_text':
        _binary_to_text_for_neusum()
    elif FLAGS.command == 'text_to_binary':
        _text_to_binary()


if __name__ == '__main__':
    tf.app.run()