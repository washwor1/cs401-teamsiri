import io
import os
import math
import tarfile
import multiprocessing
import torch
import torchaudio
import scipy
#import librosa
#import boto3
#from botocore import UNSIGNED
#from botocore.config import Config
import requests
import matplotlib
import matplotlib.pyplot as plt
#import pandas as pd
import time
from IPython.display import Audio, display
import wave
import numpy as np
import os, random


def RirConvolve(input_file, out_file, RIRpath = "RIRdataset"):

    rir = torchaudio.load(random.choice(os.listdir(RIRpath)))
    speech, _ = torchaudio.load(input_file)
    speech_ = torch.nn.functional.pad(speech, (rir.shape[1]-1, 0))
    augmented = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]
    torchaudio.save(out_file, augmented, 16000, format="wav")

prefix = "SpeechCommands/speech_commands_v0.02/"

with open("SpeechCommands/speech_commands_v0.02/convol_rir_testing.txt") as f:
    for file in f:
        input_file = prefix + file.strip("\n")
        output_file = prefix + "rir/" + file.strip("\n")
        print(input_file)
        
        RirConvolve(input_file, output_file)
        exit(0)




test = RirConvolve("output.wav")



