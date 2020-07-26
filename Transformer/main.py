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
        tensors = [torch.tensor(ex.syllable_contents, device='cuda') for ex in examples]
        results = []
        for tensor in tensors:
            tensor = tensor.reshape(-1,len(tensor))
            index1 = tensor==1
            index0 = tensor==0
            tensor[index1] = 0
            tensor[index0] = 1
            pred = model(tensor,torch.tensor([[3]]*len(tensor)).cuda())[0].sigmoid()[0][0].tolist()
            results.append(pred)
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
    "model":'transformer',
    "n_enc_vocab": 0,
    "n_dec_vocab": 0,
    "n_enc_seq": 512,
    "n_dec_seq": 512,
    "kernel_size": 3,
    "padding": 1,
    "n_layer": 3,
    "d_hidn": 768,
    "n_head": 12,
    "d_head": 64,
    "dropout": 0.4,
    "learning_rate": 5e-6,
    "batch_size":2**5,
    "weights":[3,3],
    "weight_decay":0.0001
        })
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    



#########################################################################################
def weighted_binary_cross_entropy(output, target, weights=None):
        
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def collate_fn(inputs):
    syllable_contents = [torch.tensor(x.syllable_contents) for x in inputs]
    eval_reply = torch.tensor([x.eval_reply for x in inputs]).reshape(len(inputs),1)
    enc_inputs = torch.nn.utils.rnn.pad_sequence(syllable_contents, batch_first=True, padding_value=1)

    batch = [enc_inputs,eval_reply]
    return batch
#########################################################################################
class Trainer(object):
    print(DATASET_PATH)
    TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH)
#     RAW_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH[2])
    
    def __init__(self, hdfs_host: str = None, device: str = 'cpu'):
        self.device = device
        self.task = HateSpeech(self.TRAIN_DATA_PATH, (9, 1))
#         self.raw = HateSpeech(self.RAW_DATA_PATH)
        config.n_enc_vocab = self.task.max_vocab_indexes['syllable_contents']
        config.n_dec_vocab = self.task.max_vocab_indexes['syllable_contents']
        ## train_data를 비율로 나눔
        self.model = Classification(config)
        print('can use gpu num = ',torch.cuda.device_count())
        if torch.cuda.device_count() > 1:self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.weights[0]/config.weights[1]))
        self.batch_size = config.batch_size
        self.__test_iter = None
        bind_model(self.model)
    
    def train(self):
        print('num of train data = ', len(self.task.datasets[0]))
        max_epoch = 300
        
        total_len = len(self.task.datasets[0]) ## 전체 문장 갯수
        neg_len = len([data.eval_reply for data in self.task.datasets[0] if data.eval_reply == 0])
        weight = 1. / torch.Tensor([neg_len,total_len-neg_len])
        samples_weight = torch.tensor([weight[data.eval_reply] for data in self.task.datasets[0]]) 
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, total_len)
        
        train_loader = torch.utils.data.DataLoader(self.task.datasets[0], batch_size=self.batch_size,
                                                   sampler=sampler,collate_fn=collate_fn, num_workers = 2)
        test_loader = torch.utils.data.DataLoader(self.task.datasets[1], batch_size=self.batch_size,
                                                  shuffle=False, collate_fn=collate_fn, num_workers = 2)
        
        
        optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-4, weight_decay=config.weight_decay)
        max_f1 = 0
        max_f1_i = 0
        stack = 0
        i = 0
        for epoch in range(max_epoch):
            loss_sum, acc_sum, len_batch_sum = 0., 0., 0.
            self.model.train()
            for batch in train_loader:
                self.model.zero_grad()
                data,target = batch
                data,target = data.long().cuda(),target.cuda()
                
                index1 = data==1
                index0 = data==0
                data[index1] = 0
                data[index0] = 1
                pred = self.model(data,torch.tensor([[3]]*len(data)).cuda())[0]
                loss = self.loss_fn(pred, target.float())
                
                loss.backward()
                optimizer.step()
                
                acc = torch.sum(pred.sigmoid().round()==target, dtype=torch.float32)
                acc_sum += acc.tolist()
                loss_sum += loss.tolist()
                
            print(json.dumps(
            {'type': 'train', 'dataset': 'hate_speech',
             'epoch': epoch, 'loss': loss_sum / total_len, 'acc': acc_sum / total_len}))

            loss_avg, acc_avg, f1 = self.eval(test_loader, len(self.task.datasets[1]))
            if f1>max_f1:
                max_f1 = f1
                max_f1_epoch = epoch
            print(json.dumps(
                {'type': 'test', 'dataset': 'hate_speech',
                 'loss': loss_avg,  'acc': acc_avg, 'f1': f1}))
            print('max f1 scored checkpoint : ', max_f1, max_f1_epoch)
            nsml.save(epoch)
    
    def eval(self, dataloader, total:int):
        test_loader = dataloader
        pred_lst = list()
        target_lst = list()
        loss_sum= 0.
        acc_sum = 0.
        
        self.model.eval()
        for batch in test_loader:
            data,target = batch
            data,target = data.long().cuda(),target.cuda()
            index1 = data==1
            index0 = data==0
            data[index1] = 0
            data[index0] = 1
            pred = self.model(data,torch.tensor([[3]]*len(data)).cuda())[0]
            loss = self.loss_fn(pred, target.float())
            
            acc = torch.sum(pred.sigmoid().round()==target, dtype=torch.float32)
            acc_sum += acc.tolist()
            loss_sum += loss.tolist()
            
            pred_lst+=pred.sigmoid().round().tolist()
            target_lst+=target.tolist()
            
        f1 = f1_score(pred_lst,target_lst)
        return loss_sum / total, acc_sum/ total, f1

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
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.n_output = 1
    
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
        print(config)
        trainer = Trainer(device='cuda')
        trainer.train()

