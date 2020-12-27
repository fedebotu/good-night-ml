#!/usr/bin/env python
# coding: utf-8

# # Time series predictor

import numpy as np
import time
from datetime import date
import datetime
from datetime import timedelta  
import csv
import holidays # for importing the public holidays
import re
import torch

from src.utils import *
from src.data_miner import DataMiner
from src.models import MLP
from src.dataset import GoodNightDataset


# ## Parameters
num_features = 5
min_hour = 21 # Minimum hour for sleep detection
max_hour = 5 # Maximum hour for sleep detection
train_window = 3 # Sequence length of past days
local_holidays = holidays.Italy(prov='BO') # Get the holidays in Bologna, Italy :)
EPOCHS = 500
batch_size = 16
# Directories
data_dir = "data"
data_file = "data/LastSeenDataset.csv"


# - Feature extraction: we first extract the features given the time series data of Telegram accesses.
# - Supposition: last Telegram access in very similar to the time the person goes to sleep
# ## Open Data File
with open(data_file, newline='') as csvfile:
    date_list = list(csv.reader(csvfile))

date_list = convert_to_dates(date_list)

'''Test data: search calendar for local holidays'''
print("First day is holiday: ", date_list[0][0] in local_holidays)


# ## Feature engineering
# Possible features to extract: 
# 1. Last seen time (arguably the most important)
# 2. Wake up time
# 3. Number of Telegram accesses during the previous day
# 4. Day of the week
# 5. Public holiday presence in the following day (using the holidays library)
# 6. (time spent on Telegram)
# 


data_tensor =  DataMiner(date_list).to_tensor(verbose=False)
n_features = num_features # this is number of parallel inputs
n_timesteps = train_window # this is number of timesteps


# ## Model
# Since we want to predict simple time series data, we can employ:
# - MPL: Multi Layer Perceptron, simple deep neural network with hidden layer
# - RNN: Recurrent Neural Network, more suitable for time series
# - LSTM: Long Short Term Memory, an advancement of RNN
# - Transformer: currently (2020) state of the art, but complex and possibly overpower
# - ... Other


model = MLP(n_features*n_timesteps, 1)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ## Build the Dataset
dataset = GoodNightDataset(data_file, n_timesteps)


# ## Data Augmentation
# Given that the training data is not much, we can insert some noise to augment it; this will also make the model less prone to overfitting
dataset.noisy() # apply gaussian noise


# ## Training the Model


trainloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size, shuffle=True,
                                         num_workers=0)               
model.train()
losses = []

# Training loop
for t in range(EPOCHS):
    X, y = next(iter(trainloader))
    optimizer.zero_grad()
    prediction = model.forward(X.reshape(batch_size, n_features*n_timesteps))
    loss = criterion(prediction, y)  
    loss.backward()
    optimizer.step()        
    losses.append(loss.item())
    if t%10 == 0 and t >= 10:
        print(('Epoch: {:4}  | Total mean loss: {:.6f} ').format(t, mean(losses[t-10:t])))


# ### Model evaluation on training data
# We use the trained model to predict the same data as before, this time with no noise.
# Notice that we are going to overfit if we train for too long
# Potential fixes:
# - Use validation loss
# - Use higher noise value
# - Use different noise generator
# - Find another way to augment the data
# - Collect more data
with torch.no_grad():
    for i in range(len(dataset)):
        X, y = dataset[i]
        prediction = model.forward(X.reshape(1,15)).item()
        real = y.T.item()
        print('Predicted: {:.4f} | Real: {:.4f}'.format(prediction, real))


# ## Saving the time
# We save the predicted time to send the message in a file, so that the Daemon can handle it
now = datetime.datetime.now()
seq_length = 3
with open(data_file, newline='') as csvfile:
    date_list = list(csv.reader(csvfile))
date_list = convert_to_dates(date_list)
data_tensor =  DataMiner(date_list).to_tensor(verbose=False)
X, y = create_sequences(data_tensor, seq_length)
x = get_latest_sequence(data_tensor, seq_length)

with torch.no_grad():
    p = model.forward(x.reshape(1,15)).item()
print(p)
p_sec = int(p*(max_hour+24-min_hour)*3600)
prediction = now.replace(hour=min_hour, minute=0, second=0) + timedelta(seconds=p_sec)
print('Expected time to go to sleep: ', prediction.strftime("%Y-%m-%d %H:%M:%S"))


'''Write the value on a text file to be read by the Daemon'''
with open ('data/prediction.txt','w') as z:
    z.write(prediction.strftime("%Y-%m-%d %H:%M:%S\n"))
z.close()

with open ('data/prediction_list.txt','a') as z:
    z.write(prediction.strftime("%Y-%m-%d %H:%M:%S\n"))
z.close()

