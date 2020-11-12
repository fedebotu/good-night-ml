import numpy as np
from numpy import array
import time
from datetime import date
import datetime
from datetime import timedelta  
import csv
import holidays # for importing the public holidays
import re
import torch
from statistics import mean
import sys; sys.path.append("..")
import os


class LSTM(torch.nn.Module):
    '''We use a model which should predict time series data (e.g. RNN, LSTM, Transformer)'''
    def __init__(self,n_features,seq_length, n_hidden=20, n_layers=1):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = n_hidden # number of hidden states
        self.n_layers = n_layers # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)
        
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.rand(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.rand(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()   
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)