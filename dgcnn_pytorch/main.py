#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40
from model import PointNet, ResPointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader

train_losslst = []
train_acclst = []
test_losslst = []
test_acclst = []

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')


def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=1024),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=1024),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN().to(device)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = MultiStepLR(opt, milestones=[100, 150], gamma=0.1)

    best_test_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        train_acc = 0.0
        count = 0.0
        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_acc += (preds == label).sum().item()
        outstr = 'Train %d, loss: %.6f, acc: %.6f'%(epoch, train_loss*1.0/count, train_acc*1.0/count)
        train_losslst.append(train_loss*1.0/count)
        train_acclst.append(train_acc*1.0/count)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        test_acc = 0.0
        count = 0.0
        model.eval()
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_acc += (preds == label).sum().item()
        outstr = 'Test %d, loss: %.6f, acc: %.6f'%(epoch, test_loss*1.0/count, test_acc*1.0/count)
        test_losslst.append(test_loss*1.0/count)
        test_acclst.append(test_acc*1.0/count)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model, 'checkpoints/%s/models/model.t7' % args.exp_name)
        
        io.cprint("Averate testing accuracy: " + str(sum(test_acclst)/len(test_acclst)) + '----------------------\n')



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='pointnet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    train(args, io)

    io.cprint(
        'Averate training loss: ' + str(sum(train_losslst)/len(train_losslst)))
    io.cprint("Averate training accuracy: " + str(sum(train_acclst)/len(train_acclst)))
    io.cprint("Averate testing loss: " + str(sum(test_losslst)/len(test_losslst)))
    io.cprint("Averate testing accuracy: " + str(sum(test_acclst)/len(test_acclst)))

