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
import model as gpt
from parallel import DataParallelModel, DataParallelCriterion
from torch.nn.parallel import DistributedDataParallel

from model import *
from pretrain import *
from data import HateSpeech
from sklearn.metrics import f1_score
from torchsummary import summary

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
        examples = [ex.syllable_contents for ex in examples]
        loader = torch.utils.data.DataLoader(examples, batch_size=1)
        results = []
        for data in loader:
            dec_inputs = torch.tensor(data).long().cuda()
            dec_inputs = dec_inputs.reshape(-1,len(dec_inputs))
            index1 = dec_inputs==1
            index0 = dec_inputs==0
            dec_inputs[index1] = 0
            dec_inputs[index0] = 1
            pred = model(dec_inputs)[1].max(1)[1].tolist()
            results+=pred
        return results
    nsml.bind(save=save, load=load, infer=infer)   

class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)
        
config = Config({
    "model":'gpt_transformer',
    "n_enc_vocab": 0,
    "n_dec_vocab": 0,
    "n_enc_seq": 512,
    "n_dec_seq": 512,
    "kernel_size": 3,
    "padding": 1,
    "n_layer": 5,
    "d_hidn": 768,
    "n_head": 16,
    "d_head": 64,
    "dropout": 0.1,
    "learning_rate": 5e-5,
    "batch_size":2**8,
    "lambda_":0.5,
    "weight_decay":0.0001
        })
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(inputs):
    syllable_contents = [torch.tensor(x.syllable_contents) for x in inputs]
    eval_reply = torch.tensor([x.eval_reply for x in inputs]).reshape(len(inputs),1)
    enc_inputs = torch.nn.utils.rnn.pad_sequence(syllable_contents, batch_first=True, padding_value=1)

    batch = [enc_inputs,eval_reply]
    return batch

def pretrain_collate_fn(inputs):
    syllable_contents = [torch.tensor(x.syllable_contents) for x in inputs]
    dec_inputs = torch.nn.utils.rnn.pad_sequence(syllable_contents, batch_first=True, padding_value=1)

    batch = dec_inputs
    return batch


class Trainer(object):
    print(DATASET_PATH)
    TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH[0])
    RAW_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH[1])
    
    def __init__(self, hdfs_host: str = None, device: str = 'cpu'):
        self.device = device
        self.task = HateSpeech(self.TRAIN_DATA_PATH, (9, 1))
        self.raw = HateSpeech(self.RAW_DATA_PATH)
        print('num pretrain dataset   = ',len(self.raw.datasets[0]))
        print('num train dataset      = ',len(self.task.datasets[0]))
        print('num validation dataset = ',len(self.task.datasets[1]))
        print('can use gpu num = ',torch.cuda.device_count())
        config.n_enc_vocab = self.task.max_vocab_indexes['syllable_contents']
        config.n_dec_vocab = self.task.max_vocab_indexes['syllable_contents']
        self.save_pretrain = "save_gpt_pretrain.pth"
        
    def pretrain(self):
        """ 모델 epoch 학습 """
        def train_epoch(config, epoch, model, criterion_lm, optimizer, train_loader):
            losses = []
            model.train()
            for i, value in enumerate(train_loader):
                dec_inputs = value.cuda()
                labels_lm = dec_inputs[:, 1:].contiguous()
                optimizer.zero_grad()
                index1 = dec_inputs==1
                index0 = dec_inputs==0
                dec_inputs[index1] = 0
                dec_inputs[index0] = 1
                outputs = model(dec_inputs)
                logits_lm = outputs[0]
                loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
                loss = loss_lm
                loss_val = loss_lm.item()
                losses.append(loss_val)
                loss.backward()
                optimizer.step()
            return np.mean(losses)
        pre_model = GPTPretrain(config)
        if torch.cuda.device_count() > 1:pre_model = nn.DataParallel(pre_model)
        pre_model.to(self.device)
        self.raw.datasets[0] = self.raw.datasets[0][:2000000]
        train_loader = torch.utils.data.DataLoader(self.raw.datasets[0], batch_size=config.batch_size,
                                                   shuffle=True, collate_fn = pretrain_collate_fn,
                                                   num_workers = 3,drop_last=True)
        epochs = 8
        criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=config.i_pad, reduction='mean')
        optimizer = torch.optim.AdamW(pre_model.parameters(), lr=1e-4, weight_decay = config.weight_decay)
        best_epoch, best_loss = 0, 0
        for epoch in range(epochs):
            loss = train_epoch(config, epoch, pre_model, criterion_lm, optimizer, train_loader)
            print(f'{epoch} epoch pretrain loss {loss}')
            if torch.cuda.device_count() > 1:
                pre_model.module.gpt.save(epoch, loss, self.save_pretrain)
            else:
                pre_model.gpt.save(epoch, loss, self.save_pretrain)
        torch.cuda.empty_cache()
            
    def train(self):    
        self.model = gpt.Classification(config)
        self.model.gpt.load(self.save_pretrain)
        if torch.cuda.device_count() > 1:self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        bind_model(self.model)
        
        total_len = len(self.task.datasets[0]) ## 전체 문장 갯수
        neg_len = len([data.eval_reply for data in self.task.datasets[0] if data.eval_reply == 0])
        weight = 1. / torch.Tensor([neg_len,total_len-neg_len])
        samples_weight = torch.tensor([weight[data.eval_reply] for data in self.task.datasets[0]]) 
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, total_len)
        
        train_loader = torch.utils.data.DataLoader(self.task.datasets[0], batch_size=int(config.batch_size/4), sampler=sampler,
                                                   collate_fn=collate_fn, num_workers = 3,drop_last=True)
        test_loader = torch.utils.data.DataLoader(self.task.datasets[1], batch_size=int(config.batch_size/4), shuffle=False, 
                                                  collate_fn=collate_fn, num_workers = 3,drop_last=True)
        
        train_len = len(self.task.datasets[0])
        val_len = len(self.task.datasets[1])
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate,weight_decay = config.weight_decay)
        criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=config.i_pad, reduction='mean')
        criterion_cls = torch.nn.CrossEntropyLoss()
        
        max_epoch = 100
        max_f1 = 0
        max_f1_epoch = 0
        for epoch in range(max_epoch):
            self.model.train()
            loss_sum, acc_sum, len_batch_sum = 0., 0., 0.
            for batch in train_loader:
                self.model.zero_grad()
                optimizer.zero_grad()
                data,target = batch
                dec_inputs,labels = data.long().cuda(),target.cuda()
                index1 = dec_inputs==1
                index0 = dec_inputs==0
                dec_inputs[index1] = 0
                dec_inputs[index0] = 1
                labels_lm = dec_inputs[:, 1:].contiguous()
                
                outputs = self.model(dec_inputs)
                logits_lm, logits_cls = outputs[0], outputs[1]
                
                loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
                loss_cls = criterion_cls(logits_cls, labels.squeeze())
                loss = loss_cls + config.lambda_*loss_lm
                
                loss.backward()
                optimizer.step()
                acc = torch.sum(logits_cls.max(1)[1]==labels.squeeze(), dtype=torch.float32)
                acc_sum += acc.tolist()
                loss_sum += loss_cls.tolist()
                
            print(json.dumps(
                {'type': 'train', 'dataset': 'hate_2',
                 'epoch': epoch, 'loss': loss_sum / train_len, 'acc': acc_sum / train_len}))
            nsml.save(epoch)
            self.model.eval()
            pred_lst = list()
            target_lst = list()
            loss_sum= 0.
            acc_sum = 0.
            for batch in test_loader:
                data,target = batch
                dec_inputs,labels = data.long().cuda(),target.cuda()
                
                index1 = dec_inputs==1
                index0 = dec_inputs==0
                dec_inputs[index1] = 0
                dec_inputs[index0] = 1
                outputs = self.model(dec_inputs)
                logits_lm, logits_cls = outputs[0], outputs[1]
                
                loss_cls = criterion_cls(logits_cls, labels.squeeze())
                loss = loss_cls
                
                acc = torch.sum(logits_cls.max(1)[1]==labels.squeeze(), dtype=torch.float32)
                acc_sum += acc.tolist()
                loss_sum += loss.tolist()
                
                pred_lst+=logits_cls.max(1)[1].tolist()
                target_lst+=labels.squeeze().tolist()
            f1 = f1_score(pred_lst,target_lst)
            if f1>max_f1:
                max_f1 = f1
                max_f1_epoch = epoch
            print(json.dumps(
                {'type': 'test', 'dataset': 'hate_speech',
                 'loss': loss_sum/val_len,  'acc': acc_sum/val_len, 'f1': f1}))
            print('max f1 scored checkpoint : ', max_f1, max_f1_epoch)
            torch.cuda.empty_cache()
        
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--pause', default=0)
    args = parser.parse_args()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.n_output = 2
    config.layer_norm_epsilon = 1e-12
    config.i_pad = 0
    config.d_ff = 512
    
    random.seed(2020)
    torch.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)
    np.random.seed(2020)
    
    if args.pause:
        task = HateSpeech()
        config.n_enc_vocab = task.max_vocab_indexes['syllable_contents']
        config.n_dec_vocab = task.max_vocab_indexes['syllable_contents']
        model = Classification(config)
        model.to("cuda")
        bind_model(model)
        nsml.paused(scope=locals())
    if args.mode == 'train':
        torch.cuda.empty_cache()
        print(config) 
        trainer = Trainer(device='cuda')
        trainer.pretrain()
        print('-------------------------------------pretrain complete-------------------------------------')
        trainer.train()

