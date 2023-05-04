import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import time
from IPython.display import Audio, display
import wave
import numpy as np
import os, random


max_fork = 64

labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

rir_data = ['RIRdataset/RIR_MED_7_2.wav', 'RIRdataset/RIR_MED_7_16.wav', 
            'RIRdataset/RIR_MED_7_23.wav', 'RIRdataset/RIR_ROOM_TESTED.wav']

def RirConvolve(input_file, out_file, RIRpath = "RIRdataset"):
    rir, _ = torchaudio.load(random.choice(rir_data))
    speech, _ = torchaudio.load(input_file)
    speech_ = torch.nn.functional.pad(speech, (rir.shape[1]-1, 0))
    augmented = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]
    torchaudio.save(out_file, augmented, 16000, format="wav")

prefix = "SpeechCommands/speech_commands_v0.02/"

files = ['SpeechCommands/speech_commands_v0.02/testing_list.txt', "SpeechCommands/speech_commands_v0.02/validation_list.txt"]

total_fork = 4028 + 3698
current = 0
fork_count = 0

for file in files:
    with open(file) as f:
        for file in f:

            try:
                labels.index(file.split("/")[0])
            except ValueError:
                continue
            
            fork_count += 1
            current += 1
            print(current , "/" , total_fork)
            pid = os.fork()
            if(pid == 0):
                input_file = prefix + file.strip("\n")
                output_file = prefix + file.strip("\n")
                RirConvolve(input_file, output_file)
                exit(0)
            else:
                if(fork_count >= max_fork):
                    os.wait()
                    fork_count -= 1

while(fork_count != 0):
    os.wait()
    fork_count -= 1
