from pytg import Telegram
import json
import logging
import numpy as np
from pytg.sender import Sender
from pytg.receiver import Receiver
import time
import csv
import os

logging.basicConfig(filename='logs/telegram-cli.log', filemode='a', level=logging.DEBUG)

# Configuration
polling_time = 30.0 # Seconds
data_dir = "data"
dataset = "data/LastSeenDataset.csv"
telegram_path = "/usr/bin/telegram-cli"
pubkey_path = "/home/fedebotu/tg/server.pub"


'''Create a file with the person to send the good night'''
with open('data/good_nighter.txt','r') as f:
    good_nighter = f.read() # User to send the good night wishes to
f.close()

'''
Ubuntu instructions:
Do not install via snap; it won't work. Install via:
sudo apt install telegram-cli
'''
tg = Telegram(
	telegram=telegram_path, 
	pubkey_file=pubkey_path)

last_seen = []
last_seen.append(sender.user_info(good_nighter).when) # Dict Object: parsing via
try:
    os.makedirs(data_dir)   
    print("Directory " , data_dir ,  " Created ")
except FileExistsError:
    print("Directory " , data_dir ,  " already exists. Skipping...")
    
if not os.path.exists(dataset):
    with open(dataset, 'w'):
        pass

'''Main loop for data collection'''
print("Starting main loop...")
while True:
    last = time.time()
    datum = sender.user_info(good_nighter).when # DictObject: parsing via dot structure
    # If the last seen is new, store the datum
    if last_seen[-1] != datum: 
        last_seen.append(datum)
        with open(dataset,'a') as file:
            writer = csv.writer(file)
            writer.writerow([str(datum)])
        print("Last seen: ", last_seen)
    # Use sleep function instead of an if so to avoid useless CPU runtime consumption
    time.sleep(polling_time)