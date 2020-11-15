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
from src.models import LSTM

'''Parameters'''
num_features = 5
min_hour = 21 # Minimum hour for sleep detection
max_hour = 5
train_window = 3 # Sequence length
local_holidays = holidays.Italy(prov='BO') # Get the holidays in Bologna, Italy :)
train_episodes = 500
batch_size = 3
# Variables
data_dir = "data"
dataset = "data/LastSeenDataset.csv"

'''Feature engineering
Possible features to extract: 
1. Last seen time (arguably the most important)
2. Wake up time
3. Number of Telegram accesses during the previous day
4. Day of the week
5. Public holiday presence in the following day (using the holidays library)
6. (time spent on Telegram)'''

with open(dataset, newline='') as csvfile:
    date_list = list(csv.reader(csvfile))
date_list = convert_to_dates(date_list)

'''Test data: search calendar for local holidays'''
print("First day is holiday: ", date_list[0][0] in local_holidays)

data_tensor =  DataMiner(date_list).to_tensor(verbose=False)
print(data_tensor)

'''TO BE IMPLEMENTED: DATA AUGMENTATION'''

''' We use the last 3 elements trend to predict the time series
Credits: https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
The sequence on which we have a prediction is the last train_window days'''
X, y = create_inout_sequences(data_tensor, train_window)
X = X.transpose(0, 2, 1)

n_features = num_features # this is number of parallel inputs
n_timesteps = train_window # this is number of timesteps

# convert dataset into input/output

# create LSTM neural network
mv_net = LSTM(n_features,n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-1)

mv_net.train()

# Training loop
for t in range(train_episodes):
    for b in range(0,len(X),batch_size):
        inpt = X[b:b+batch_size,:,:]
        target = y[b:b+batch_size]
        x_batch = torch.tensor(inpt,dtype=torch.float32)    
        y_batch = torch.tensor(target,dtype=torch.float32)  
        mv_net.init_hidden(x_batch.size(0))
        output = mv_net(x_batch) 
        loss = criterion(output, y_batch)  
        loss.backward()
        optimizer.step()        
        optimizer.zero_grad()
    if t%10:
        print(('Step: {:4}   |   Loss: {:.6f} ').format(t, loss.item()))

# TEST
#with torch.no_grad():
#    print('Predicted:', mv_net(torch.tensor(X[4:7,:,:],dtype=torch.float32))[0])
#    print('Real:', y[4])

'''Replace prediction time with today's date'''
now = datetime.datetime.now()
with torch.no_grad():
    p = mv_net.forward(torch.tensor(X[-batch_size-1:-1,:],dtype=torch.float32))[0].numpy()
p_sec = int(p[0]*(max_hour+24-min_hour)*3600)
prediction = now.replace(hour=min_hour, minute=0, second=0) + timedelta(seconds=p_sec)
print('Expected time to go to sleep: ', prediction.strftime("%Y-%m-%d %H:%M:%S"))

'''Write the value on a text file to be read by the Daemon'''
with open ('prediction.txt','w') as z:
    z.write(prediction.strftime("%Y-%m-%d %H:%M:%S\n"))
z.close()

'''Write the same value in the log file'''
with open ('logs/prediction_list.txt','a') as z:
    z.write(prediction.strftime("%Y-%m-%d %H:%M:%S\n"))
z.close()
