import numpy as np
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
print(os.getcwd())

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

class DataMiner():
    '''We create a data tensor ready to be fed into the neural network for predictions'''
    def __init__(self, data, local_holidays= holidays.Italy(prov='BO'), MIN_H=21, MAX_H=5, MAX_SLEPT=12, MAX_ACCESSES=100, NUM_FEATURES=5):
        # Last seen during this time will be considered as last seen time
        # BEWARE: cannot detect times out of this range
        self.data = data
        self.local_holidays = local_holidays
        self.MIN_H, self.MIN_M, self.MIN_S = MIN_H, 0, 0
        self.MAX_H, self.MAX_M, self.MAX_S = MAX_H, 0, 0
        self.MAX_SLEPT = MAX_SLEPT # In hours
        self.MAX_ACCESSES = MAX_ACCESSES
        self.NUM_FEATURES = NUM_FEATURES # Number of features we choose to use
        self.fall_asleep_datetime = []
        self.wake_up_datetime = []
        self.fall_asleep = [] # array containing seconds from MIN_XX time to go to bed
        self.time_slept = []
        self.day_of_week = []
        self.holiday_presence = []
        self.num_accesses = []
        # Suppose detection started before the minimum hour (MIN_H)
        self.min_asleep_time = data[0][0]
        self.min_asleep_time = self.min_asleep_time.replace(hour= self.MIN_H, minute= self.MIN_M, second= self.MIN_S)
        self.max_asleep_time = data[0][0]
        self.max_asleep_time = self.max_asleep_time.replace(day= data[0][0].day, hour= self.MAX_H, minute= self.MAX_M, second= self.MAX_S) + timedelta(days=1)
        self.saved = True
        self.n_accesses = 0
    
    def _fill_outlier(self, sleeptime):
        """Replace data by their averages"""
        self.fall_asleep_datetime.append(self.fall_asleep_datetime[-1] + timedelta(days=1)) # Add same sleeptime as yesterday with + 1 day
        self.wake_up_datetime.append(self.wake_up_datetime[-1] + timedelta(days=1)) # Add same wakeup as yesterday with + 1 day
        self.fall_asleep.append(mean(self.fall_asleep))
        self.time_slept.append(mean(self.time_slept))
        self.day_of_week.append(self.fall_asleep_datetime[-1].weekday())
        self.holiday_presence.append(self.fall_asleep_datetime[-1] in self.local_holidays)
        self.num_accesses.append(mean(self.num_accesses))
        #print('Outlier found and replaced!')
    
    def _save_data(self, d, n_accesses, sleeptime):
        '''
        Fill in the blanks--Let's check if there are no new data: it means that the data was either not collected,
        or it is an outlier: hence, let's set the new sleeptime to be the same as the last recorded one
        '''
        sleeptime_s = time_diff_sec(sleeptime, self.min_asleep_time)
        time_slept_s = time_diff_sec(d, sleeptime)
        self.fall_asleep_datetime.append(sleeptime)
        self.wake_up_datetime.append(d)
        self.fall_asleep.append(sleeptime_s)
        self.time_slept.append(time_slept_s)
        self.day_of_week.append(d.weekday())
        self.holiday_presence.append(d in self.local_holidays)
        self.num_accesses.append(n_accesses)
    
    def _print_raw_data(self):
        print("Fall asleep time: ", self.fall_asleep_datetime)
        print("Wake up time: ", self.wake_up_datetime)
        print("Fall asleep time (seconds from MIN): ", self.fall_asleep)
        print("Time slept (seconds): ", self.time_slept)
        print("Day of the week (from 0 = Monday): ", self.day_of_week)
        print("Holiday presence: ", self.holiday_presence)
        print("Number of accesses: ", self.num_accesses)
        
    def _print_matrix(self, matrix):
        print(matrix)
        
    def _to_array(self, data):
        return np.array(data).reshape(-1, 1)

    def _build_matrix(self, asleep, slept, day, holiday, num_accesses):
        '''We preprocess the data so that they are "normalized" to a fixed scale (for easy time recovery)'''
        num_observations = len(asleep)
        asleep = self._to_array(asleep) /  ((self.MAX_H+24-self.MIN_H)*3600 + (self.MAX_M-self.MIN_M)*60 + (self.MAX_S-self.MIN_S))
        slept = self._to_array(slept) / (self.MAX_SLEPT*3600)
        day = self._to_array(day) / 6
        holiday = self._to_array(holiday) # no need to normalize,  true/false
        accesses = self._to_array(num_accesses)/ self.MAX_ACCESSES
        return np.concatenate((asleep, slept, day, holiday, accesses)).reshape(self.NUM_FEATURES, num_observations)

    def to_tensor(self, verbose=True):
        '''Main loop for sweeping through the data and building the array'''
        for i in range(len(self.data)):
            # If time is a candidate for falling asleep (=good night) then save it
            d = self.data[i][0]
            if d > self.min_asleep_time and d < self.max_asleep_time:
                sleeptime = d # save this time
                self.saved = False
            else:
                if len(self.fall_asleep_datetime) == 0:
                    if not self.saved:
                        self._save_data(d, self.n_accesses, sleeptime) # First datum is not empty, of course
                        self.min_asleep_time = self.min_asleep_time + timedelta(days=1) # d.replace(hour= MIN_H, minute= MIN_M, second= MIN_S)
                        self.max_asleep_time = self.max_asleep_time + timedelta(days=1) #d.replace(day= d.day, hour= MAX_H, minute= MAX_M, second= MAX_S) + timedelta(days=1)
                else:
                    if (d - self.wake_up_datetime[-1] > timedelta(days=1, hours = self.MAX_SLEPT)):
                        self._fill_outlier(sleeptime)
                        d = (self.fall_asleep_datetime[-1] + timedelta(days=1))
                        self.min_asleep_time = self.min_asleep_time + timedelta(days=1) # d.replace(hour= MIN_H, minute= MIN_M, second= MIN_S)
                        self.max_asleep_time = self.max_asleep_time + timedelta(days=1) #d.replace(day= d.day, hour= MAX_H, minute= MAX_M, second= MAX_S) + timedelta(days=1)
                    if not self.saved:
                        self._save_data(d, self.n_accesses, sleeptime)
                        self.min_asleep_time = self.min_asleep_time + timedelta(days=1) # d.replace(hour= MIN_H, minute= MIN_M, second= MIN_S)
                        self.max_asleep_time = self.max_asleep_time + timedelta(days=1) #d.replace(day= d.day, hour= MAX_H, minute= MAX_M, second= MAX_S) + timedelta(days=1)
                self.saved = True 
                self.n_accesses = 0
            self.n_accesses += 1 # increase counter for every different last seen ~ number of accesses

        matrix = self._build_matrix(self.fall_asleep, self.time_slept, self.day_of_week, self.holiday_presence, self.num_accesses)
        
        if verbose:
            print('Raw data from the distribution:\n')
            self._print_raw_data()
            print('Matrix created from raw data:\n')
            print(matrix)
            
        return torch.from_numpy(matrix)