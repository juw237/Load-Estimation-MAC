# -*- coding:utf-8 -*-
"""

"""
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from tqdm import tqdm
from scipy.signal import savgol_filter

from get_data import device, get_mape,get_rmse, get_R2,setup_seed
from models import Seq2Seq
setup_seed(20)

def seq2seq_test(args, Dte, path, m, n):
    # Make sure m,n are only from Training dataset, no data leakage!!
    # Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    if args.task=='prediction':
        output_size = args.output_size
    else:
        output_size = args.seq_len
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))  #this is used to flatten a nested list
        y.extend(target)
        #seq (bacth_size:128, sequence_length:24, input_length:7)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
    #Transfer back to origional scale
    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('Test Dataset mape:', get_mape(y, pred))
    print('Test Dataset rmse:', get_rmse(y, pred))
    print('Test Dataset r2:', get_R2(y, pred))

    return y, pred

# prediction mean with confidence interval
def seq2seq_test_100(args, Dte, path, m, n):
    # Make sure m,n are only from Training dataset, no data leakage!!
    # Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    pred_all = []
    y_final = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for i in range(100):
        pred=[]
        y=[]
        for (seq, target) in tqdm(Dte):
            target = list(chain.from_iterable(target.data.tolist()))  #this is used to flatten a nested list
            y.extend(target)
            #seq (bacth_size:128, sequence_length:24, input_length:7)
            seq = seq.to(device)
            with torch.no_grad():
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                pred.extend(y_pred)
        #Transfer back to origional scale
        y, pred = np.array(y), np.array(pred)
        pred = (m - n) * pred + n
        y = (m - n) * y + n
        pred_all.append(pred)
    mean_vals, lower, upper = confidence_interval(pred_all)
    print('Test Dataset mape for 100 times: ', get_mape(y, mean_vals))
    print('Test Dataset rmse for 100 times: ', get_rmse(y, mean_vals))
    print('Test Dataset R2 100 times: ', get_R2(y, mean_vals))

    return y, mean_vals, lower, upper

#Visulize Performance on Test set
def plot_all_diagnal(args,y, pred):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    pred_size = args.output_size
    plt.scatter(y[::pred_size], pred[::pred_size], c='green')
    # plt.scatter(y, pred, c='green')

    Min_lim=min(min(y[::pred_size]),min(pred[::pred_size]))-5
    Max_lim=max(max(y[::pred_size]),max(pred[::pred_size]))+5
    plt.plot(np.linspace(Min_lim, Max_lim, 20), np.linspace(Min_lim, Max_lim, 20), c="red", marker='.', linestyle=':')
    plt.xlabel('True Observation')
    plt.ylabel('Prediction')
    path = os.path.abspath(os.path.dirname(os.getcwd()))
    plt.savefig(path+'/results/ Diagnal Plot.png' )
    plt.show()

def plot_one(args,y,pred,num):
    # plot
    pred_size=args.output_size
    plt.figure(figsize=(8, 5))
    plt.plot(y[(num-1)*pred_size:num*pred_size], label='GroundTruth')
    plt.plot(pred[(num-1)*pred_size:num*pred_size], label='Prediction')
    plt.xlabel("No of Steps")
    plt.ylabel("Parameter Values")
    plt.title("One example of prediction on test dataset")
    plt.legend()

    plt.show()


def plot_onestep_all(args, y, pred,t_test):
    # plot
    if args.task=='prediction':
        pred_size = args.output_size
        input_size=args.seq_len
        case_num=int(len(pred)/args.output_size)
        #fix length
        t_test=t_test.iloc[input_size:input_size+case_num]
    if args.task == 'reconstruction':
        pred_size = args.seq_len
        case_num = int(len(pred)/args.seq_len)
        t_test = t_test.iloc[0:case_num]

    t_test['True_1']=y[::pred_size]
    t_test['Pred_1']=pred[::pred_size]
    t_test[args.Time_column_name] = pd.to_datetime(t_test[args.Time_column_name])
    plt.figure(figsize=(16, 5))
    # plt.plot(y[::pred_size], label='True Observation')
    # plt.plot(pred[::pred_size], label='Prediction')
    plt.plot(t_test[args.Time_column_name],t_test['True_1'], label='True Observation')
    plt.plot(t_test[args.Time_column_name], t_test['Pred_1'], label='Prediction')
    plt.xlabel("Time")
    plt.ylabel("Parameter Values")
    plt.title("1 Step Prediction on Test Dataset")
    plt.legend()
    path = os.path.abspath(os.path.dirname(os.getcwd()))
    plt.savefig(path+'/results/ 1 Step Prediction Plot.png')
    plt.show()

def plot_RMSE_all(args, y, pred,t_test):
    # plot
    if args.task == 'prediction':
        pred_size = args.output_size
        input_size = args.seq_len
        case_num = int(len(pred) / args.output_size)
        # fix length
        t_test = t_test.iloc[input_size:input_size + case_num]
    if args.task == 'reconstruction':
        pred_size = args.seq_len
        case_num = int(len(pred) / args.seq_len)
        t_test = t_test.iloc[0:case_num]
    t_test[args.Time_column_name] = pd.to_datetime(t_test[args.Time_column_name])
    plt.figure(figsize=(16, 5))
    RMSE = []
    for i in range(0, len(y), pred_size):
        segment_true = y[i:i + pred_size]
        segment_pred = pred[i:i + pred_size]
        RMSE.append(get_rmse(segment_true, segment_pred))
    t_test['RMSE'] = RMSE
    t_test['5_step_MA'] = t_test['RMSE'].rolling(window=10).mean()
    window_length = 40# must be odd
    polyorder = 5
    t_test['SG_filtered_RMSE'] = savgol_filter(t_test['RMSE'], window_length, polyorder)

    plt.plot(t_test[args.Time_column_name] ,t_test['RMSE'], label='Root Mean Squared Error')
    # plt.plot(t_test['time'], t_test['5_step_MA'], label='MA_RMSE',color='red')
    plt.plot(t_test[args.Time_column_name], t_test['SG_filtered_RMSE'], label='SG_RMSE',color='red')
    plt.xlabel("Time")
    plt.ylabel("RMSE")
    plt.title("%i Step(s) Prediction RMSE Values" % args.output_size)
    plt.legend()
    path = os.path.abspath(os.path.dirname(os.getcwd()))
    plt.savefig(path+'/results/ RMSE Plot.png')
    plt.show()

    t_test.to_csv(path+'/results/Test RMSE.csv')

def plot_MAPE_all(args, y, pred):
    # plot
    pred_size = args.output_size
    # plt.figure(figsize=(8, 5))
    MAPE=[]
    for i in range(0, len(y), pred_size):
        segment_true = y[i:i + pred_size]
        segment_pred = pred[i:i + pred_size]
        MAPE.append(get_mape(segment_true, segment_pred))
    plt.plot(MAPE, label='Mean Absolute Percentage Error')
    plt.title("MAPE for each prediction on test dataset")
    plt.show()


def plot_R2_all(args, y, pred):
    # plot
    pred_size = args.output_size
    # plt.figure(figsize=(8, 5))
    R2 = []
    for i in range(0, len(y), pred_size):
        segment_true = y[i:i + pred_size]
        segment_pred = pred[i:i + pred_size]
        R2 .append(get_R2(segment_true, segment_pred))
    plt.plot(R2, label='R Square')
    plt.title("R2 for each prediction on test dataset")
    plt.show()

def plot_one_with_confidence(args,y,pred,lower,upper, num):
    # plot
    pred_size=args.output_size
    plt.figure(figsize=(20, 5))
    plt.plot(y[(num-1)*pred_size:num*pred_size], label='GroundTruth')
    plt.plot(pred[(num-1)*pred_size:num*pred_size], label='Prediction_mean')
    plt.plot(lower[(num - 1) * pred_size:num * pred_size], label='Prediction_lower')
    plt.plot(upper[(num - 1) * pred_size:num * pred_size], label='Prediction_upper')
    plt.title("1 Step Prediction on Test Dataset")
    plt.legend()
    plt.show()

def confidence_interval(data, confidence=0.95):
    # Calculate the mean and standard deviation for each position
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)

    # Calculate the margin of error for each position
    z_score = 1.96
    margin_error = z_score * (std / np.sqrt(len(data)))

    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    return mean, lower_bound, upper_bound


