import copy
import os
import sys
from itertools import chain

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import get_data
import pandas as pd
import numpy as np
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device= torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device='cpu'

from models import Seq2Seq
import pickle

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

#用测试集后avaliable的输入预测一次未来值（理论上这些预测没有真实值对照）
def predict_one_step(args,path,m,n,seq_len,num):
    #data = get_data.load_data('data.csv')
    data = get_data.load_data('Volve_F10_30s_smoothed_TVT_SG_filter_1.csv')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])

    train = data[:int(len(data) * 0.7)]
    test = data[int(len(data) * 0.7):len(data)]
    #Normalization using origional(Training) fitted scaler
    list_1 = test.columns[2:]
    df1 = test.iloc[:, 0:2]
    # with open('minmax_scaler.pkl', 'rb') as f:
    #     scaler = pickle.load(f)
    # load the scaler
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    #scaler = MinMaxScaler()
    #scaler.fit(data.iloc[:, 2:])
    df2 = scaler.transform(test.iloc[:, 2:])
    df2 = pd.DataFrame(df2, columns=list_1)
    #index are messed up, need reset
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df3 = pd.concat([df1, df2], axis=1)
    test_normalized=df3

    def process(dataset):
        #所有target variable 使用之前train 的scalar normalization
        load = dataset[dataset.columns[1]]
        load = (load - n) / (m - n)
        load = load.tolist()
        dataset = dataset.values.tolist()
        #
        seq = []
        #数据集最后的 seq_len 个
        for i in range(len(test) - seq_len, len(test) - seq_len+1):
            pred_seq = []
            #train_label = []
            for j in range(i, i + seq_len):
                x = [load[j]]
                #####modify!!!
                for c in range(2, 12):
                    x.append(dataset[j][c])
                pred_seq.append(x)

            #for j in range(i + seq_len, i + seq_len + num):
                #train_label.append(load[j])

            pred_seq = torch.FloatTensor(pred_seq)
            #train_label = torch.FloatTensor(train_label).view(-1)
            seq.append(pred_seq)

        seq = get_data.MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        return seq
    # 开始预测
    pred = []
    seq=process(test_normalized)
    for seq in seq:
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
            # y_pred为一个列表，长度为12
    pred = np.array(pred)
    return pred * (m - n) + n
