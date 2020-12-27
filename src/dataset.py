import torch
from torch.utils.data import Dataset
import csv
import re
from src.utils import convert_to_dates
from src.data_miner import DataMiner

class GoodNightDataset(Dataset):
    """Build Dataset for easy interfacing with PyTorch"""
    def __init__(self, data_root, seq_length):
        self.seq_length = seq_length
        with open(data_root, newline='') as csvfile:
            date_list = list(csv.reader(csvfile))
        date_list = convert_to_dates(date_list)
        self.data =  DataMiner(date_list).to_tensor(verbose=False)
        self.n_features = self.data.shape[0]
        # the sequence on which we have a prediction is the last train_window days
        self.X, self.y = self.create_sequences()
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def _apply_noise(self, x, scale, noise):
        """Crop noisy datum to be in [0,1]"""
        noisy = x + scale*noise(1)
        if min(noisy.item(), 1) == 1:
            noisy = torch.tensor(1)
        if max(noisy.item(), 0 ) == 0:
            noisy = torch.tensor(0)
        return noisy

    def create_sequences(self, data_type=torch.float32):
        '''We create a list of training data divided in inputs X and outputs y'''
        X = []
        y = []
        L = self.data.shape[1]
        tw = self.seq_length
        for i in range(L-tw):
            train_seq = torch.zeros(self.n_features, tw)
            for j in range(self.n_features):
                train_seq[j]= self.data[j][i:i+tw]
            train_label = self.data[0][i+tw:i+tw+1] 
            X.append(train_seq)
            y.append(train_label)
        return torch.transpose(torch.stack(X), 2, 1).type(data_type), torch.stack(y).type(data_type)

    def get_latest_sequence(self, data_type=torch.float32):
        '''Get latest sequence for making prediction'''
        X = []
        tw = self.seq_length
        idx = self.data.shape[1] # index of the last element
        seq = torch.zeros(self.n_features, tw)
        for j in range(self.n_features):
            seq[j]= self.data[j][idx-tw:idx]
        X.append(seq)
        return torch.transpose(
            torch.stack(X), 2, 1).type(data_type)


    def noisy(self, scale=0.05, noise=torch.randn):
        """Build batch with noise"""
        for i in range(self.X.shape[0]):
            #y[i] = self._apply_noise(y[i], scale, noise)
            for j in range(self.X.shape[1]):
                for k in [0, 1, 4]: # don't put noise on day of the week and festive day presence
                    self.X[i, j, k] = self._apply_noise(self.X[i, j, k], scale, noise)         
        return