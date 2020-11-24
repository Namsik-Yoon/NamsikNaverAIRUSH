#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import typing
from typing import List, Dict, Tuple
from pydoc import locate
from collections import Counter
from random import shuffle
import random
import os

from torchtext.data import Dataset, Example, Field, LabelField
from torchtext.vocab import Vocab
import torch
from nsml.constants import DATASET_PATH

## e.g. {"syllable_contents": [3, 32, 218, 12, 25, 2, 205, 337, 16, 2, 113, 9, 2, 558, 195, 16, 2, 113, 17, 68, 2, 288, 51, 39, 12, 25, 4], "eval_reply": 0}

random.seed(2020)
torch.manual_seed(2020)

class HateSpeech(object):
    MAX_LEN = 512
    UNK_TOKEN = 0  # '<unk>'
    PAD_TOKEN = 1  # '<pad>'
    SPACE_TOKEN = 2  # '<sp>'
    INIT_TOKEN = 3  # '<s>'
    EOS_TOKEN = 4  # '<e>'
    TOKENS = [PAD_TOKEN, UNK_TOKEN, SPACE_TOKEN, INIT_TOKEN, EOS_TOKEN]
    FIELDS_TOKEN_ATTRS = ['init_token', 'eos_token', 'unk_token', 'pad_token']
    FIELDS_ATTRS = FIELDS_TOKEN_ATTRS + ['sequential', 'use_vocab', 'fix_length']

    VOCAB_PATH = 'fields.json'

    def __init__(self, corpus_path=None, split: Tuple[int, int] = None):
        self.fields, self.max_vocab_indexes = self.load_fields(self.VOCAB_PATH)

        if corpus_path:
            self.examples = self.load_corpus(corpus_path)
            ## self.exmaples = [첫번째 문장의 ex, 두번째 문장의 ex ...]
            ## ex = 토큰화된 문장과 label 포함
            
            if split:
                shuffle(self.examples)
                total = len(self.examples) ## 전체 문장의 수
                pivot = int(total / sum(split) * split[0])
                self.datasets = [Dataset(self.examples[:pivot], fields=self.fields),
                                 Dataset(self.examples[pivot:], fields=self.fields)]
#                 print('train count : ',len(self.datasets[0]))
#                 print('validation count : ',len(self.datasets[1]))
            else:
                self.datasets = [Dataset(self.examples, fields=self.fields)]
    def load_corpus(self, path) -> List[Example]:
        """
        self.examples를 만듦
        """
        if 'raw' in path:
            path = '/data/hate_raw/train/raw.json'
        preprocessed = []
        with open(path) as fp:
            for line in fp:
                if line: ## line = {"syllable_contents": [3, 128, 200, 5, 30, 30, 268, 2, 130, 5, 69, 6, 6, 6, 6, 4], "eval_reply": 0}
                    ex = Example()
                    ## preprocessed리스트안에 ex를 부여
                    for k, v in json.loads(line).items():
                        setattr(ex, k, v)
                        ## ex.syllable_contents = [3, 128, 200, 5, 30, 30, 268, 2, 130, 5, 69, 6, 6, 6, 6, 4]
                        ## ex.eval_reply = 0
                        """새로 넣은 내용(3개 이상 중복되는 문자는 제거)"""
                        ## [1,1,1,2,2,3,3,3] -> [1,1,2,2,3,3]
                        ##########################################################################
                        inputs = ex.syllable_contents
                        max_len = len(inputs)
                        result = inputs.copy()
                        for i in range(max_len-2):
                            check_tokens = inputs[i:i+3]
                            if (check_tokens[0]==check_tokens[1]) and (check_tokens[0]==check_tokens[2]):
                                for j in range(i+2,max_len):
                                    if check_tokens[0] == inputs[j]:
                                        result[i+2] = 'temp'
                        ex.syllable_contents = [x for x in result if x != 'temp']
                        ###############################################################################
#                         ex.syllable_contents = ex.syllable_contents[1:-1]
                        index0 = [i for i,value in enumerate(ex.syllable_contents) if value == 0]
                        index1 = [i for i,value in enumerate(ex.syllable_contents) if value == 1]
                        for idx in index0:ex.syllable_contents[idx] = 1
                        for idx in index1:ex.syllable_contents[idx] = 0
                    preprocessed.append(ex)
        return preprocessed

    def dict_to_field(self, dicted: Dict) -> Field:
        field = locate(dicted['type'])(dtype=locate(dicted['dtype']))
        for k in self.FIELDS_ATTRS:
            setattr(field, k, dicted[k])

        if 'vocab' in dicted:
            v_dict = dicted['vocab']
            vocab = Vocab()
            vocab.itos = v_dict['itos']
            vocab.stoi.update(v_dict['stoi'])
            vocab.unk_index = v_dict['unk_index']
            if 'freqs' in v_dict:
                vocab.freqs = Counter(v_dict['freqs'])
        else:
            vocab = Vocab(Counter())
            field.use_vocab = False
        field.vocab = vocab

        return field

    def load_fields(self, path) -> Dict[str, Field]:
        """
        self.fields, self.max_vocab_indexes를 만듦
        """
        
        loaded_dict = json.loads(open(path).read())
        max_vocab_indexes = {k: v['max_vocab_index'] for k, v in loaded_dict.items()}
        return {k: self.dict_to_field(v) for k, v in loaded_dict.items()}, max_vocab_indexes

