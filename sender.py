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
from src.utils import *
from pytg import Telegram
import json
import logging # Useful for debugging
logging.basicConfig(filename='logs/telegram-cli.log', filemode='a', level=logging.DEBUG)
import numpy as np
from pytg.sender import Sender
from pytg.receiver import Receiver


'''
Sender daemon: our goal is now to build a daemon which, given the prediction, will send the good night message at that time or a bit earlier (we can decide to send it earlier to be more confident about the result
'''

polling_time = 300 # seconds to wait
sent = False # We start giving the false condition for sent
advance_time = 30 # Minutes we send the message in advance with respect to the prediction for being more confident the receiver will get the goodnight promptly
telegram_path = "/usr/bin/telegram-cli"
pubkey_path = "/home/fedebotu/tg/server.pub"

'''Create a file with the person to send the good night'''
with open('data/good_nighter.txt','r') as f:
    good_nighter = "Federico_Berto" # str(f.read()) # User to send the good night wishes to
f.close()
print(good_nighter)

'''
Ubuntu instructions:
Do not install via snap; it won't work. Install via:
sudo apt install telegram-cli
'''
tg = Telegram(
	telegram=telegram_path, 
	pubkey_file=pubkey_path)

receiver = Receiver(host="localhost", port=4458)
sender = Sender(host="localhost", port=4458)

with open('data/prediction.txt', 'r') as f:
    # convert to string
    prediction = datetime.datetime.strptime(f.read(), "%Y-%m-%d %H:%M:%S\n" ) 
f.close()

print('Starting main loop...')
while True:
    """
    We read the messages and store them in an array. UTF-8
    encoding is important for including emojis
    """

    messages = []
    for m in enumerate(open('data/messages.txt', 'r', encoding='utf-8')):
        messages.append(m[1])
    f.close()

    '''If we pass the prediction time, then we send a message and wait until the next prediction has come out '''
    if(not sent and prediction < (datetime.datetime.now() - timedelta(minutes=advance_time))):
        sender.send_msg(good_nighter, choose_message(messages)) # To be substituted with the good nighter
        sent = True
        print("Message has been sent")
    
    with open('data/prediction.txt', 'r') as f:
        # convert to string
        old = prediction
        prediction = datetime.datetime.strptime(f.read(), "%Y-%m-%d %H:%M:%S\n")
        if(prediction != old): sent=False # if we have a new prediction, let's send it when the time condition is met
    f.close()
    
    print('Next prediction: ',  prediction)
    time.sleep(polling_time)
