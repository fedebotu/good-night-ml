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

# Transform all the elements in the dataset into date objects and sort them
def convert_to_dates(dates):
    for i in range(len(dates)):
        dates[i][0] = datetime.datetime.strptime(dates[i][0], "%Y-%m-%d %H:%M:%S" ) 
    return sorted(dates)


# Measure time difference in seconds
def time_diff_sec(t_f, t_0):
    return (t_f - t_0).total_seconds()

def to_array(data):
    return np.array(data).reshape(-1, 1)

# def create_inout_sequences(dt, tw):
#     inout_seq = []
#     L = dt.shape[1]
#     for i in range(L-tw):
#         train_seq = torch.zeros(NUM_FEATURES, tw)
#         for j in range(NUM_FEATURES):
#             train_seq[j]= dt[j][i:i+tw]
#         train_label = dt[0][i+tw:i+tw+1]
#         inout_seq.append((train_seq ,train_label))
#     print(inout_seq)
#     return inout_seq

'''We create a list of training data divided in inputs X and outputs y'''
def create_inout_sequences(dt, tw, n_features=5):
    X = []
    y = []
    L = dt.shape[1]
    for i in range(L-tw):
        train_seq = torch.zeros(n_features, tw)
        for j in range(n_features):
            train_seq[j]= dt[j][i:i+tw]
        train_label = dt[0][i+tw:i+tw+1] 
        X.append(train_seq.numpy()); y.append(train_label.numpy())
    return array(X), array(y)

'''Get latest sequence for making prediction'''
def get_latest_sequence(dt, tw, n_features=5):
    X = []
    idx = dt.shape[1]-1 # index of the last element
    for i in range(tw):
        seq = torch.zeros(n_features, tw)
        for j in range(n_features):
            seq[j]= dt[j][idx-tw:idx]
        X.append(seq.numpy())
    return array(X)

'''Choose a message from the txt file'''
import random as rnd
def choose_message(messages,random=True, number=0):
    if random:
        m = rnd.sample(messages, k=1)[0]
    else: m = messages[number]
    return m