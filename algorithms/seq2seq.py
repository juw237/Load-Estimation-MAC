# -*- coding:utf-8 -*-
import os
import sys

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
#pandas warning level
import pandas as pd
pd.options.mode.chained_assignment = None

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from get_data import nn_seq_mo
from predict import predict_one_step
from args import seq2seq_args_parser
from model_train import seq2seq_train
from model_test import seq2seq_test,seq2seq_test_100
from model_test import plot_all_diagnal,plot_one,plot_one_with_confidence,plot_onestep_all,plot_MAPE_all,plot_RMSE_all,plot_R2_all

path = os.path.abspath(os.path.dirname(os.getcwd()))
Seq2Seq_LSTM_PATH = path + '/models/seq2seq.pkl'

if __name__ == '__main__':
    args = seq2seq_args_parser()#Seeq2Seq-LSTM
    flag = 'seq2seq'

    dataset_name ='petrobras_Well_A.csv'
    Target=['torque']
    Predictor=[
        # 'time',
        # 'torque',
        'rotary_speed',
        # 'ROP',
        'block_position',
        'weight_on_hook',
        'weight_on_bit',
        # 'SPP',
        'fluid_flow',
        'Total pump stoke',
        # 'total_pit_volume',
        'heave_bit_depth',
        'hole_depth_filtered',
        # 'choke_pressure_filtered',
        # 'kill_pressure_filtered',
        # 'trip_tank_volume'
        'bottom_rotary_speed_filtered',
        'inclination_filtered'
    ]
    if args.include_target:
        args.input_size=len(Predictor)+1
    else:
        args.input_size =len(Predictor)

    Dtr, Val, Dte, m, n,t_train,t_val,t_test = nn_seq_mo(dataset_name,Target,Predictor, args.include_target,args.Train_p, args.Valid_p,args,start=0,end=0.14,resample=args.resample,freq=args.freq,flter_SG=args.Filter_SG,window_length=20,polyorder=5,seq_len=args.seq_len, B=args.batch_size, num=args.output_size)
    seq2seq_train(args, Dtr, Val, Seq2Seq_LSTM_PATH)
    y_test,pred_test=seq2seq_test(args, Dte, Seq2Seq_LSTM_PATH, m, n)
    #y,pred,lower,upper=seq2seq_test_100(args, Dte, LSTM_PATH, m, n)

    #Visulization
    plot_all_diagnal(args,y_test,pred_test)
    # plot_one(args, y_test, pred_test, 1)

    # plot_MAPE_all(args, y_test, pred_test,t_test)
    plot_RMSE_all(args, y_test, pred_test,t_test)
    # plot_R2_all(args, y_test, pred_test,t_test)

    plot_onestep_all(args, y_test, pred_test,t_test)
    # plot_one_with_confidence(args, y, pred_test, lower, upper, 10)

    # #make last 1 prediction
    # pred_1=predict_one_step(args,Seq2Seq_LSTM_PATH, m, n, seq_len=args.seq_len, num=args.output_size)
    # print(pred_1)
    # #make any 1 prediction