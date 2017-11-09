# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf
from gensim.models import Word2Vec

model = Word2Vec.load_word2vec_format('../../models/GoogleNews-vectors-negative300.bin.gz', binary=True)
word2vec_dim = 300

ner_cls = {}

def load_ner_cls(ners):
    for i, ner in enumerate(set(ners)):
        ner_cls[ner] = i 

def load(data_path):
    train_path = os.path.join(data_path, "astro.train.txt")
    valid_path = os.path.join(data_path, "astro.valid.txt")
    test_path = os.path.join(data_path, "astro.test.txt")

    train_data = AstroData(_read_data(train_path))
    valid_data = AstroData(_read_data(valid_path))
    test_data = AstroData(_read_data(test_path))
    
    load_ner_cls(train_data.ner)
    
    for data in [train_data, valid_data, test_data]:
        data.numerize_ner_cls()

    return train_data, valid_data, test_data

def _read_data(filename):
    """right now the input file will temporarily be *-ner.tsv, i.e., in the format of
    w11*\t*ner11 (first sentence, word and ner is split by \t)
    w12*\t*ner12
    ...
    
    w21*\t*ner21 (second sentence)
    ...
    
    the output is 
    [
        [
            [w11, ner11], [w12, ner12],...
        ],
        [
            [w21, ner21], [w22, ner22],...
        ],...
    ]
    """
    with tf.gfile.GFile(filename, "r") as f:
        data = f.read().decode("utf-8").split('\n\n')
        data = [l.split('\n') for l in data]
        data = [[w.split('\t') for w in l] for l in data]
        return data

class AstroData(object):
    
    """input is output of _read_data"""

    def __init__(self, data, model, word2vec_dim):
        for utt in data:
            words, ners = [], []
            for word, ner in utt:
                word_vec = AstroData._word2vec(word, model, word2vec_dim)
                if not word_vec: continue
                words.append(word_vec)
                ners.append(AstroData.get_ner(ner, iob=False))
            self.utterance.append(words)
            self.ners.append(ners)
                    
#         self.classification
#         self.intent
    
    @classmethod
    def get_ner(cls, ner, iob=False):
        if iob: return ner
        else: return ner if ner == 'O' else ner[2:]
    
    def numerize_ner_cls(self):
        self.ner = [AstroData._ner_to_id(n) for n in self.ner]        
    
#     def word2vec(self, model, word2vec_dim):
#         self.utterance = [[AstroData._word2vec(w, model) for w in utt] for utt in self.utterance]
    
    @classmethod
    def _word2vec(cls, word, model, word2vec_dim):
        if word in model.wv.vocab:
            return model.wv[word]
        else:
            print("word %s not found in model." % (word))
            return [0.] * word2vec_dim

    @classmethod
    def _ner_to_id(cls, ner):
        try:
            return ner_cls[ner]
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise


class AstroInput():
    
    def __init__(self, astro_data):
        self.data = astro_data
    
    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen
    
def astro_producer(X, y, batch_size, max_steps, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.
        name: the name of this operation (optional).

    Returns:
        X and y, each shaped [batch_size, max_steps] with padding zero. 
        seqlen: shaped [batch_size], sequence lengths

    Raises:
        tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    
    with tf.name_scope(name, "AstroProducer", [X, y, batch_size, max_steps]):
        X = tf.convert_to_tensor(X, name="raw_input", dtype=tf.float32)
        y = tf.convert_to_tensor(y, name="raw_label", dtype=tf.int32)

        data_len = tf.size(X)
        batch_len = data_len // batch_size
        data = tf.reshape(X[0 : batch_size * batch_len],
                                            [batch_size, batch_len])

        epoch_size = (batch_len - 1) // max_steps
        assertion = tf.assert_positive(
                epoch_size,
                message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * max_steps],
                                                 [batch_size, (i + 1) * max_steps])
        x.set_shape([batch_size, max_steps])
        y = tf.strided_slice(data, [0, i * max_steps + 1],
                                                 [batch_size, (i + 1) * max_steps + 1])
        y.set_shape([batch_size, max_steps])
        return X, y
