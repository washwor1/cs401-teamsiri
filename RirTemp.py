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

#torchaudio.sox_effects.init_sox_effects()
#waveform, sample_rate = torchaudio.load("original.wav") 
#torchaudio.save("output.wav", waveform, sample_rate, encoding="PCM_S", bits_per_sample=16, format="wav")
#torchaudio.save(buffer_, waveform, sample_rate, format="wav")

SAMPLE_RIR_PATH = "wavFiles/dataRIR.wav"#"bedTest.wav"
SAMPLE_WAV_SPEECH_PATH = "bedTest.wav"
SAMPLE_WAV_PATH = SAMPLE_WAV_SPEECH_PATH

def _get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.extend([
      ["lowpass", f"{resample // 2}"],
      ["rate", f'{resample}'],
    ])
  return torchaudio.load(path)
  #return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


def get_speech_sample(*, resample=None):
  return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)

def get_sample(*, resample=None):
  return _get_sample(SAMPLE_WAV_PATH, resample=resample)

def get_rir_sample(*, resample=None, processed=False):
  rir_raw, sample_rate = _get_sample(SAMPLE_RIR_PATH, resample=resample)
  if not processed:
    return rir_raw, sample_rate
  #rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]
  #rir = rir / torch.norm(rir, p=2)
  #rir = torch.flip(rir, [1])
  return rir, sample_rate
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=True)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=True)

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")


######################################
sample_rate = 16000#16000 #8000

#rir_raw, sample_rate = get_rir_sample(resample=sample_rate)
rir_raw, _ = get_rir_sample(resample=sample_rate)

plot_waveform(rir_raw, sample_rate, title="Room Impulse Response (raw)", ylim=None)
plot_specgram(rir_raw, sample_rate, title="Room Impulse Response (raw)")
#play_audio(rir_raw, sample_rate)

#rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]

rir = rir_raw
rir = rir / torch.norm(rir, p=2)
rir = torch.flip(rir, [1])
#print(rir[0])
rir[0] = rir[0] * 0.5
#print(rir[0])
#print(rir.shape)

#print_stats(rir)
plot_waveform(rir, sample_rate, title="Room Impulse Response", ylim=None)
######################################

speech, _ = get_speech_sample(resample=sample_rate)

##speech, sample_rate = get_speech_sample(resample=sample_rate)

speech_ = torch.nn.functional.pad(speech, (rir.shape[1]-1, 0))
print(speech_.shape)
augmented = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]


augmented = torch.nn.functional.conv1d(speech[None, ...], rir[None, ...])[0]
plot_waveform(speech, sample_rate, title="Original", ylim=None)
plot_waveform(augmented, sample_rate, title="RIR Applied", ylim=None)

plot_specgram(speech, sample_rate, title="Original")
play_audio(speech, sample_rate)

plot_specgram(augmented, sample_rate, title="RIR Applied")
#play_audio(augmented, sample_rate)

#torchaudio.save("output.wav", speech, sample_rate, encoding="PCM_S", bits_per_sample=16, format="wav")
#torchaudio.save("outputAugmented.wav", augmented, sample_rate, encoding="PCM_S", bits_per_sample=16, format="wav")
torchaudio.save("output.wav", speech, sample_rate, format="wav")
torchaudio.save("outputAugmentedRIR.wav", augmented, sample_rate, format="wav")


print("end")