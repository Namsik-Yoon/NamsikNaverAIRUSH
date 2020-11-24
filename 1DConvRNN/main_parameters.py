#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import json
import math
import typing
from typing import Dict, List
import os
from argparse import ArgumentParser
import random

from torch import nn, optim
from torch.nn import functional as F
import torch
from torchtext.data import Iterator
from tqdm import tqdm
import numpy as np
import nsml
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
from torchtext.data import Example

from model import BaseLine
from data import HateSpeech
from sklearn.metrics import f1_score

def bind_model(model):
    def save(dirname, *args):
        if torch.cuda.device_count() > 1:
            checkpoint = {'model': model.module.state_dict()}
        else:
            checkpoint = {'model': model.state_dict()}
        torch.save(checkpoint, os.path.join(dirname, 'model.pt'))

    def load(dirname, *args):
        checkpoint = torch.load(os.path.join(dirname, 'model.pt'))
        model.load_state_dict(checkpoint['model'])

    def infer(raw_data, **kwargs):
        model.eval()
        examples = HateSpeech(raw_data).examples
        tensors = [torch.tensor(ex.syllable_contents, device='cuda').reshape([-1, 1]) for ex in examples]
        results = [model(ex).sigmoid().tolist()[0][0] for ex in tensors]
        return results
    
    nsml.bind(save=save, load=load, infer=infer)

######################################################################################################################
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
######################################################################################################################
#########################################################################################
def weighted_binary_cross_entropy(output, target, weights=None):
        
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))
#########################################################################################
class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

config = Config({
    "model":'base_line',
    "hidden_dim": 512,
    "filter_size": 5,
    "padding":2,
    "dropout": 0.4,
    "embedding_dim":256,
    "learning_rate": 1e-3,
    "adam_epsilon": 1e-8,
    "warmup_steps": 0,
    "batch_size":2**7,
    "weights":[3,10]
        })
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH)

    def __init__(self, hdfs_host: str = None):
        self.device = config.device
        self.task = HateSpeech(self.TRAIN_DATA_PATH,[9,1])
        self.model = BaseLine(config.hidden_dim, config.filter_size,config.dropout, 
                              self.task.max_vocab_indexes['syllable_contents'], config.embedding_dim,config.padding) ## 저장모델도 같게 해줘~
        ## Baseline : self.model = BaseLine(256, 3, 0.2, self.task.max_vocab_indexes['syllable_contents'], 384)
#         print('can use gpu num = ',torch.cuda.device_count())
        if torch.cuda.device_count() > 1:self.model = nn.DataParallel(self.model)
        self.model.to(config.device)
        self.loss_fn = nn.BCEWithLogitsLoss(torch.tensor(config.weights[0]/config.weights[1]))
        self.batch_size = config.batch_size
        self.__test_iter = None
        bind_model(self.model)

    @property
    def test_iter(self) -> Iterator:
        if self.__test_iter:
            self.__test_iter.init_epoch()
            return self.__test_iter
        else:
            self.__test_iter = Iterator(self.task.datasets[-1], batch_size=self.batch_size, repeat=False,
                                        sort_key=lambda x: len(x.syllable_contents), train=False,
                                        device=self.device)
            return self.__test_iter

    def train(self):
        max_epoch = 50
        optimizer = optim.Adam(self.model.parameters(),lr=config.learning_rate)
        total_len = len(self.task.datasets[0])
        ds_iter = Iterator(self.task.datasets[0], batch_size=self.batch_size, repeat=False,
                           sort_key=lambda x: len(x.syllable_contents), train=True, device=self.device)
        min_iters = 10
        max_acc = 0
        max_acc_epoch = 0
        for epoch in range(max_epoch):
            loss_sum, acc_sum, len_batch_sum = 0., 0., 0.
            ds_iter.init_epoch()
            tr_total = math.ceil(total_len / self.batch_size)

            self.model.train()
            for batch in ds_iter:
                self.model.zero_grad()
                data = batch.syllable_contents.cuda()
                target = batch.eval_reply.reshape(len(batch.eval_reply),1).cuda()
                pred = self.model(data).cuda()
                loss = self.loss_fn(pred, target)
                
                loss.backward()
                optimizer.step()
                
                acc = torch.sum(pred.sigmoid().round()==target, dtype=torch.float32)
                
                len_batch = len(batch)
                len_batch_sum += len_batch
                acc_sum += acc.tolist()
                loss_sum += loss.tolist() * len_batch
            
            pred_lst, loss_avg, acc_sum, te_total = self.eval(self.test_iter, len(self.task.datasets[1]))
            acc_current = acc_sum / te_total
            if acc_current>max_acc:
                max_acc = acc_current
                max_acc_epoch = epoch
            nsml.save(epoch)
        print(f'max accuracy = {max_acc} when epoch {max_acc_epoch}')

    def eval(self, iter:Iterator, total:int) -> (List[float], float, List[float], int):
        pred_lst = list()
        target_lst = list()
        loss_sum= 0.
        acc_sum = 0.

        self.model.eval()
        for batch in iter:
            data = batch.syllable_contents.cuda()
            target = batch.eval_reply.reshape(len(batch.eval_reply),1).cuda()
            
            pred = self.model(data).cuda()
            
            accs = torch.sum(pred.sigmoid().round()==target, dtype=torch.float32)
            losses = self.loss_fn(pred, target)
            
            pred_lst += pred.sigmoid().round().squeeze().tolist()
            acc_sum += accs.tolist()
            target_lst += batch.eval_reply.tolist()
            loss_sum += losses.tolist() * len(batch)
        return pred_lst, loss_sum / total, acc_sum, total

    def save_model(self, model, appendix=None):
        file_name = 'model'
        if appendix:
            file_name += '_{}'.format(appendix)
        torch.save({'model': model, 'task': type(self.task).__name__}, file_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--pause', default=0)
    args = parser.parse_args()
    
    torch.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)
    np.random.seed(2020)
    random.seed(2020)
    
    if args.pause:
        task = HateSpeech()
        model = BaseLine(config.hidden_dim, config.filter_size,config.dropout,
                         task.max_vocab_indexes['syllable_contents'], config.embedding_dim,config.padding)
        model.to("cuda")
        bind_model(model)
        nsml.paused(scope=locals())
    if args.mode == 'train':
        print(config)
        trainer = Trainer()
        trainer.train()
        print('-'*50)
        ##############################################
        config.embedding_dim = 128
        print(config)
        trainer = Trainer()
        trainer.train()
        print('-'*50)
        ##############################################
        config.embedding_dim = 512
        print(config)
        trainer = Trainer()
        trainer.train()
        config.embedding_dim = 256
        print('-'*50)
        ##############################################
        config.weights = [10,3]
        print(config)
        trainer = Trainer()
        trainer.train()
        print('-'*50)
        ##############################################
        config.weights = [1,1]
        print(config)
        trainer = Trainer()
        trainer.train()

