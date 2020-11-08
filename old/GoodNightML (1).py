#!/usr/bin/env python
# coding: utf-8

# # Good Night Machine Learning
# Telegram autosender learning when to send a good night message to your loved ones


from pytg import Telegram
import json
import logging # Useful for debugging
import numpy as np
from pytg.sender import Sender
from pytg.receiver import Receiver
import time
import csv
import os
# Enable verbose debugging
# logging.basicConfig(level=logging.DEBUG)
good_nighter = "Eleonora_Morselli" # User to send the good night wishes to
root_path = "/home/pi/Documents/GoodNightML/"
dataset = root_path + "data/LastSeenDataset.csv"
telegram_dir = "/home/pi/tg/bin/telegram-cli"
pubkey_dir = "/home/pi/tg/server.pub"
data_dir= root_path + "data"

# Configuration
polling_time = 60.0 # Seconds



'''
Ubuntu instructions:
Do not install via snap; it won't work. Install via:
sudo apt install telegram-cli
'''
tg = Telegram(
	telegram=telegram_dir,
	pubkey_file=pubkey_dir)


receiver = Receiver(host="localhost", port=4458)
sender = Sender(host="localhost", port=4458)

#Test
#sender.send_msg("Federico_Berto", "Hello World!")



last_seen = []
last_seen.append(sender.user_info(good_nighter).when) # Dict Object: parsing via
print("Last seen: ", last_seen)


try:
    os.makedirs(data_dir)   
    print("Directory " , data_dir ,  " Created ")
except FileExistsError:
    print("Directory " , data_dir ,  " already exists. Skipping...")
 
if not os.path.exists(dataset):
    with open(dataset, 'w'):
        pass
'''
with open(dataset,'a') as file:
    writer = csv.writer(file)
    writer.writerow([str(last_seen[-1])])
'''


# Main loop for data collection
#last = time.time()
# print(last)
print("Starting main loop...")
while True:
    '''
   if time.time() - last > polling_time: # Polling only in a certain interval of time
    '''
    last = time.time()
    datum = sender.user_info(good_nighter).when # DictObject: parsing via dot structure
    # If the last seen is new, store the datum
    if last_seen[-1] != datum: 
        last_seen.append(datum)
        with open(dataset,'a') as file:
            writer = csv.writer(file)
            writer.writerow([str(datum)])
        print("Last seen data updated: ", last_seen)
    # Use sleep function instead of an if so to avoid useless CPU runtime consumption
    time.sleep(polling_time) 



