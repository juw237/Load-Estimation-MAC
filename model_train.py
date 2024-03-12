# -*- coding:utf-8 -*-
"""

"""
import copy
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np

from get_data import device, setup_seed
from models import Seq2Seq

setup_seed(20)

print('Device=',device)

#pytorch可以给我们提供两种方式来切换训练和评估(推断)的模式，分别是：model.train() 和 model.eval()。
#一般用法是：在训练开始之前写上 model.trian() ，在测试时写上 model.eval() 。
def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for (seq, label) in Val:
        #验证集 我们只是想看一下训练的效果，并不是想通过验证集来更新网络时，就可以使用with torch.no_grad()
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)

def seq2seq_train(args, Dtr, Val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    batch_size = args.batch_size
    # model.to(device)
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
    loss_function = nn.MSELoss().to(device)
    #Dedfine Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
    #
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = [] #当前 epoch下训练集loss
        for (seq, label) in Dtr: #训练集
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad() #前一个batch的梯度计算结果，没有保留的必要了。所以在下一次梯度更新的时候，先使用optimizer.zero_grad把梯度信息设置为0。
            loss.backward()  #Pytorch的autograd就会自动沿着计算图反向传播，计算每一个叶子节点的 梯度
            optimizer.step() #optimizer.step用来更新参数

        scheduler.step() #一个epoch结束，更新学习率
        # validation
        val_loss = get_val_loss(args, model, Val) #当前 epoch下测试集loss
        #Save the best model
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()
        # pytorch可以给我们提供两种方式来切换训练和评估(推断)的模式，分别是：model.train() 和 model.eval()。
        # 一般用法是：在训练开始之前写上 model.trian() ，在测试时写上 model.eval() 。
    #保存训练过程中在验证集上表现最好的模型
    state = {'models': best_model.state_dict()}
    torch.save(state, path)

