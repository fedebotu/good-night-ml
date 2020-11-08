import datetime
from datetime import timedelta
import numpy as np

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