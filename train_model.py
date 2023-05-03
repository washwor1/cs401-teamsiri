import matplotlib.pyplot as plt
import IPython.display as ipd
import torch.optim as optim
from tqdm import tqdm
import torchaudio
import argparse
import os
import tensorflow as tf
#local files
import audioModel
import importDataset
import pertubation
import librosa
import wave
import numpy as np
import torch
import audioModel as am
from torch.utils.mobile_optimizer import optimize_for_mobile


all_labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',\
          'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', \
          'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

batch_size = 256
origin_frequency = 16000
new_frequency = 8000
n_epoch = 20
log_interval = 20
model_path = "test.ptf"
save_model_file = None #"models/.ptf" <-- Format for file or whatever
save_model_file_app = None #"models/.ptl"

def get_target_label(old_label):
    mappedIndex = {"yes":   1, "no":    0,
                   "up":    3, "down":  2,
                   "left":  5, "right": 4,
                   "on":    7, "off":   6,
                   "stop":  9, "go":    8}
    return mappedIndex[old_label]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device_type():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device_info(device):
    if str(device) == "cuda":
        return 1, True
    else:
        return 0, False

def set_tranform_function(origin_frequency, new_frequency, device):
    transformFunction = torchaudio.transforms.Resample(orig_freq=origin_frequency, new_freq=new_frequency)
    return transformFunction.to(device)

if(save_model_file == None or save_model_file_app == None):
    print("Save model files are not set")
    exit(1)

device = get_device_type()
num_workers, pin_memory = get_device_info(device)
print("Device: " + str(device))
print("Worker: " + str(num_workers))
print("Memory: " + str(pin_memory))

print("Getting training, testing sets")
#get training and testing set
train_set = importDataset.SubsetSC("training")
test_set = importDataset.SubsetSC("testing")

print("Getting Transform function")
#get transform function
transform = set_tranform_function(origin_frequency, new_frequency, device)

print("Getting loaders")
#get testing and training loaders --> Will have to look at what this actually is!
train_loader = importDataset.getTrainLoader(train_set, batch_size, True, num_workers, pin_memory)
test_loader = importDataset.getTestLoader(test_set, batch_size, False, False, num_workers, pin_memory)

print("Creating new model")
model = audioModel.M5()

print("Setting optimizer")
optimizer = audioModel.setOptimizer(model, learn_rate = 0.01, weight_decay=0.0001)
print("Setting scheduler")
# reduce the learning after 20 epochs by a factor of 10
scheduler = audioModel.setScheduler(optimizer, step_size=10, gamma=0.1)

pbar_update = 1 / (len(train_loader) + len(test_loader))

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        audioModel.train(model, epoch, log_interval, train_loader, device, transform, optimizer, pbar, pbar_update)
        audioModel.test(model, epoch, test_loader, device, transform, pbar, pbar_update)
        scheduler.step()

#Save the model

model = model.to('cpu')
torch.save(model.state_dict(), save_model_file)
example_input = torch.randn(1, 1, 16000)
# Convert the PyTorch model to TorchScript
traced_script_module = torch.jit.trace(model, example_input)
optimized = optimize_for_mobile(traced_script_module)
# Save the TorchScript model
traced_script_module._save_for_lite_interpreter(save_model_file_app)

