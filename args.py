# -*- coding:utf-8 -*-
import argparse
import torch

# seq2seq
def seq2seq_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='input dimension')
    parser.add_argument('--include_target', type=bool, default=False, help='whether target variable is considered as a predictor')  # number of predictors
    parser.add_argument('--task', type=str, default='prediction', help='prediction or reconstruction')  # 'prediction' | 'reconstruction'(include_target must = False!)

    parser.add_argument('--input_size', type=int, default=5, help='input dimension')  # number of predictors, will change automatically
    parser.add_argument('--seq_len', type=int, default=6, help='seq len')   #Input time series steps
    parser.add_argument('--output_size', type=int, default=1, help='output dimension') #predict time series steps
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')    #Hidden size of each LSTM Layer
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')   #Number of LSTM Layers
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer') #SGD adam
    # parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu")) #If Nvidia Cuda
    # parser.add_argument('--device', default=torch.device("mps" if torch.backends.mps.is_available() else "cpu")) If Mac GPU (MPS)
    parser.add_argument('--device', default=torch.device("cpu")) #If MAC CPU.

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='LSTM direction')
    parser.add_argument('--Train_p', type=int, default=0.4, help='Train %')  # Training Data percentage
    parser.add_argument('--Valid_p', type=int, default=0.1, help='Validation %')  # Validation Data percentage
    parser.add_argument('--Test_p', type=int, default=0.5, help='Test %')  # Test Data percentage
    parser.add_argument('--Filter_SG', type=bool, default=True, help='apply SG Filter')  # True means apply SG filter
    parser.add_argument('--resample', type=bool, default=False, help='downsample data')  # True means apply re-sample filter
    parser.add_argument('--freq', type=str, default='30s',help='resample frequency')

    parser.add_argument('--step_size', type=int, default=5, help='step size')   # learning rate scheduler 的 step size
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--Time_column_name', type=str, default='Time s', help='Time_column_name')

    args = parser.parse_args()

    return args
