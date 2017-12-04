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
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

model = KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin.gz', binary=True)
word2vec_dim = 300

ner_cls = {}
maxlen = 0

def load_ner_cls(ner_lists):
    ners = [ner for ners in ner_lists for ner in ners]
    for i, ner in enumerate(set(ners)):
        ner_cls[ner] = i 
        
def get_maxlen(utt_lists):
    return max([len(utt) for utt in utt_lists])

def load_from_path(data_path):
    train_path = os.path.join(data_path, "astro_tmp.train.tsv")
    valid_path = os.path.join(data_path, "astro_tmp.valid.tsv")
    test_path = os.path.join(data_path, "astro_tmp.test.tsv")

    train_data = AstroData(_read_data(train_path), model, word2vec_dim)
    valid_data = AstroData(_read_data(valid_path), model, word2vec_dim)
    test_data = AstroData(_read_data(test_path), model, word2vec_dim)
    
    load_ner_cls(train_data.ners + valid_data.ners + test_data.ners)
    global maxlen
    maxlen = get_maxlen(train_data.utterances + valid_data.utterances + test_data.utterances)
    
    for data in [train_data, valid_data, test_data]:
        data.numerize_ner_cls()
        data.pad_seqlen(maxlen)

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
        data = [[w.split('\t') for w in l] for l in data if len(l) > 0]
        return data

class AstroData(object):
    
    """input is output of _read_data"""

    def __init__(self, data, model, word2vec_dim):
        self.utterances = []
        self.ners = []
        self.seqlen = []
        for utt in data:
            words, ners = [], []
            for word, ner in utt:
                word_vec = AstroData._word2vec(word, model, word2vec_dim)
#                 if not word_vec: continue
                words.append(word_vec)
                ners.append(AstroData.get_ner(ner, iob=False))
            self.utterances.append(words)
            self.ners.append(ners)
            self.seqlen.append(len(words))
                    
#         self.classification
#         self.intent
    
    @classmethod
    def get_ner(cls, ner, iob=False):
        if iob: return ner
        else: return ner if ner == 'O' else ner[2:]
    
    def numerize_ner_cls(self):
        self.ners = [[self._ner_to_id(n) for n in ner] for ner in self.ners]        
    
#     def word2vec(self, model, word2vec_dim):
#         self.utterance = [[AstroData._word2vec(w, model) for w in utt] for utt in self.utterance]
    
    @classmethod
    def _word2vec(cls, word, model, word2vec_dim):
        if word in model.wv.vocab:
            return model.wv[word]
        else:
#             print("word %s not found in model." % (word))
            return np.zeros(word2vec_dim, dtype=np.float32)

    def _ner_to_id(self, ner):
        try:
            classes = [0] * self.num_ner_classes()
            classes[ner_cls[ner]] = 1
            return classes
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
    
    def num_ner_classes(self):
        return len(ner_cls)
    
    def pad_seqlen(self, maxlen):
        self.utterances = [self._pad_seqlen(utt, maxlen) for utt in self.utterances]
    
    def _pad_seqlen(self, seq, maxlen):
        padzeros = [np.zeros(word2vec_dim, dtype=np.float32) for _ in xrange(maxlen - len(seq))]
        return seq + padzeros
        
class AstroInput():
    
    def __init__(self, astro_data):
        self.utt = astro_data.utterances
        self.ners = astro_data.ners
        self.seqlen = astro_data.seqlen
        self.num_ner_classes = astro_data.num_ner_classes()
        self.batch_id = 0
        self.maxlen = maxlen
        self.dim = word2vec_dim
    
    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        
        if self.batch_id == len(self.utt):
            self.batch_id = 0
        batch_data = self.utt[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.utt))]
        batch_labels = self.ners[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.ners))]
        batch_labels = [e for sublist in batch_labels for e in sublist]
        batch_seqlen = self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.seqlen))]
        batch_mask = np.zeros((batch_size * self.maxlen))
        for i, l in enumerate(batch_seqlen):
            batch_mask[i * self.maxlen : i * self.maxlen + l] = 1
        self.batch_id = min(self.batch_id + batch_size, len(self.utt))
        return batch_data, batch_labels, batch_seqlen, batch_mask


def load():
    train_data, valid_data, test_data = load_from_path('./')
    trainset = AstroInput(train_data)
    validset = AstroInput(valid_data)
    testset = AstroInput(test_data)
    return trainset, validset, testset
