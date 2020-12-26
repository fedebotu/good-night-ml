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
from torch import nn
from statistics import mean
import sys; sys.path.append("..")
import os

class MLP(nn.Module):
    """Multi-Layer Perceptron: this is the 
    simplest Deep Neural Network model
    https://www.kaggle.com/pinocookie/pytorch-simple-mlp"""
    def __init__(self, input_dim, output_dim, hidden_layer=100):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer = hidden_layer
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_layer),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, self.output_dim),
            #nn.Sigmoid() # use for constraining the output to [0,1]
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.reshape(1,self.input_dim)
        x = self.layers(x)
        return x
    
    
class LSTM(torch.nn.Module):
    '''We use a model which should predict time series data (e.g. RNN, LSTM, Transformer)'''
    def __init__(self,n_features,seq_length, n_hidden=1000, n_layers=1):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = n_hidden # number of hidden states
        self.n_layers = n_layers # number of LSTM layers (stacked)
    
        self.l_lstm = nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = nn.Linear(self.n_hidden*self.seq_len, 1)
        
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
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)

    
class RNN(nn.Module):
    """https://www.kaggle.com/kanncaa1/recurrent-neural-network-with-pytorch"""
    def __init__(self, input_dim, output_dim, hidden_dim=100, layer_dim=1):
        super(RNN, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out
    
    
class SimpleRNN(nn.Module):
    """https://gist.github.com/spro"""
    def __init__(self, hidden_size=10):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        self.inp = nn.Linear(1, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
        self.out = nn.Linear(hidden_size, 1)

    def step(self, input, hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, 1))
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        return outputs, hidden