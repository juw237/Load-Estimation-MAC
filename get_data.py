# -*- coding:utf-8 -*-

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error,r2_score
from torch.utils.data import Dataset, DataLoader
from pickle import dump
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device= torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device ='cpu'
import pickle

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True #lock seed


def load_data(file_name):
    """
    :return: dataframe
    """
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/' + file_name
    df = pd.read_csv(path, encoding='gbk')
    #df.fillna(df.mean(), inplace=True)

    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


# Multiple outputs data processing.
# seq_len----输入时序长度
# B-----batch size
# num---- 预测时序长度
def nn_seq_mo(dataname,Target,Predictor, include_target,Train_percent,Valid_percent,args,start,end,resample,freq,flter_SG,window_length,polyorder,seq_len, B, num):
    data = load_data(dataname)
    args.Time_column_name = data.columns[0]

    start_idx = int(data.shape[0] * start)
    end_idx = int(data.shape[0] * end)
    data = data.iloc[start_idx:end_idx]
    # data = data1.copy()
    data=data.reset_index(drop=True)

    #Data Trime and resample
    if resample:
        data[ args.Time_column_name] = pd.to_datetime(data[ args.Time_column_name])
        data = data.set_index( args.Time_column_name)
        data = data.resample(freq).mean()
        data = data.reset_index(drop=True)
    #SG filter
    if flter_SG:
        # window_length = 20 # Should be odd
        # polyorder = 5  # Polynomial order
        for col in data.columns[1:]:
            data[col] = savgol_filter(data[col], window_length, polyorder)
    #Add visulization!!!

    #re-arange dataframe as 'Time,Target,Predictors'
    first_col_name = data.columns[0]
    df1=pd.concat([data[[first_col_name]],data[Target]],axis=1)
    # Minmax_scaler to normalize predictors
    scaler = MinMaxScaler()
    scaler.fit(data[Predictor])
    # save the scaler for later use.
    dump(scaler, open('scaler.pkl', 'wb'))
    #
    df2 = scaler.transform(data[Predictor])
    df2 = pd.DataFrame(df2, columns=Predictor)
    df3 = pd.concat([df1, df2], axis=1)
    data=df3
    Predictor_num = data.shape[1]

    #Default Train 80%   Valid 10%   Test 10%
    train = data.iloc[:int(len(data) * Train_percent)]
    val = data.iloc[int(len(data) * Train_percent):int(len(data) * (Train_percent+Valid_percent))]
    test = data.iloc[int(len(data) * (Train_percent+Valid_percent)):len(data)]
    #Record time
    t_train=train[[ args.Time_column_name]]
    t_val= val[[ args.Time_column_name]]
    t_test= test[[ args.Time_column_name]]

    #Gwt Max an Min of Target variable.!!!!!!!!!!!!
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])
    # m=315
    # n=300

    def process(dataset, batch_size, step_size, shuffle):
        if args.task=='prediction':
            #step_size----每次取值要空几个data point.
            #Min/Max scaler normalize the target variable.
            load = dataset[dataset.columns[1]] #target variable
            load = (load - n) / (m - n)
            load = load.tolist()
            dataset = dataset.values.tolist()
            #
            seq = []
            for i in range(0, len(dataset) - seq_len - num, step_size):  # len(dataset) - seq_len - num 是一个数据段中可用的样本数量
                train_seq = []
                train_label = []

                for j in range(i, i + seq_len):
                    x = [load[j]]
                    if include_target:
                        for c in range(2, Predictor_num):
                            x.append(dataset[j][c])
                        train_seq.append(x)
                    else:
                        for c in range(3, Predictor_num):
                            x.append(dataset[j][c])
                        train_seq.append(x)
                for j in range(i + seq_len, i + seq_len + num):
                    train_label.append(load[j])
                # transfer data into torch
                train_seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)  #flaten the label tensor
                seq.append((train_seq, train_label))
        else:# re-construction
            # step_size----每次取值要空几个data point.
            # Min/Max scaler normalize the target variable.
            load = dataset[dataset.columns[1]]  # target variable
            load = (load - n) / (m - n)
            load = load.tolist()
            dataset = dataset.values.tolist()
            #
            seq = []
            for i in range(0, len(dataset)-seq_len, step_size):  # len(dataset) 是一个数据段中可用的样本数量
                train_seq = []
                train_label = []

                for j in range(i, i + seq_len):
                    x = [load[j]]
                    for c in range(3, Predictor_num):
                        x.append(dataset[j][c])
                    train_seq.append(x)
                for j in range(i , i + seq_len ):
                    train_label.append(load[j])
                # transfer data into torch
                train_seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)  # flaten the label tensor
                seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
        # shuffle (bool, optional) – set to True to have the data reshuffled at every epoch
        # drop_last (bool, optional) – set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
        return seq

    Dtr = process(train, B, step_size=1, shuffle=True)
    Val = process(val, B, step_size=1, shuffle=True)
    Dte = process(test, B, step_size=1, shuffle=False)

    return Dtr, Val, Dte, m, n,t_train,t_val,t_test
def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))

def get_rmse(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: rmse
    """
    return mean_squared_error(x,y,squared = False)
def get_R2(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: rmse
    """
    return r2_score(x,y)