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
from scipy.io import wavfile

batch_size = 256
origin_frequency = 16000
new_frequency = 8000
model_path = "models/98_raw.ptf"

labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

def get_target_label(old_label):
    mappedIndex = {"yes":   1, "no":    0,
                   "up":    3, "down":  2,
                   "left":  5, "right": 4,
                   "on":    7, "off":   6,
                   "stop":  9, "go":    8}
    return mappedIndex[old_label]

def get_device_type():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device_info(device):
    if str(device) == "cuda":
        return 1, True
    else:
        return 0, False

def set_tranform_function(origin_frequency_t, new_frequency_t, device):
    transformFunction = torchaudio.transforms.Resample(orig_freq=origin_frequency_t, new_freq=new_frequency_t)
    return transformFunction.to(device)

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


# print("Setting optimizer")
# optimizer = audioModel.setOptimizer(model, learn_rate = 0.01, weight_decay=0.0001)
# print("Setting scheduler")
# # reduce the learning after 20 epochs by a factor of 10
# scheduler = audioModel.setScheduler(optimizer, step_size=10, gamma=0.1)

waveform, *_ = train_set[0]

attack_model = pertubation.M5(n_input=waveform.shape[0], n_output=10)
attack_model.load_state_dict(torch.load(model_path))
attack_model = attack_model.to(device)
test_loader_iterator = iter(test_loader)

currentBatch = 0
pertubation_results = []
batch_number = 0
correct_label = []
pertubated_label = []

while True:
    try:
        #extract batches of wave_forms, and target labels
        data, target = next(test_loader_iterator)
        
        #changes target values to be what we want based on the utterance of current wave_form
        for i in range(currentBatch, currentBatch + data.shape[0]):
            wave_form, sample_rate, utterance, *_ = test_set[i]
            correct_label.append(utterance)
            currentTarget = i % 256
            new_label_index = get_target_label(utterance)
            target[currentTarget] = new_label_index

        currentBatch += 256
        data, target = data.to(device), target.to(device)
        pertubation_results.append(pertubation.attack(attack_model, device, data, target, targeted=True))

        for p in range(0, pertubation_results[batch_number].shape[0]):
            new_prediction = audioModel.predict(pertubation_results[batch_number][p], attack_model, device, transform, importDataset.index_to_label, perform_transform=False)
            pertubated_label.append(new_prediction)

        batch_number += 1
        print("batch: " + str(batch_number))
    except StopIteration:
        print("end")
        break
    

#Will make this look cleaner, but for now it's fine
mismatches = 0
# left_count = 0
# right = 0

correct_pertubated_dict = {
    "yes":  ["no", 0, 0],
    "no":   ["yes", 0, 0],
    "on":   ["off", 0, 0],
    "off":  ["on", 0, 0],
    "right":["left", 0, 0],
    "left": ["right", 0, 0],
    "down": ["up", 0, 0],
    "up":   ["down", 0, 0],
    "stop": ["go", 0, 0],
    "go":   ["stop", 0, 0]
}

for i in range(0, len(pertubated_label)):
    correct_pertubated_dict[correct_label[i]][2] += 1
    if(correct_pertubated_dict[correct_label[i]][0] == pertubated_label[i]):
        correct_pertubated_dict[correct_label[i]][1] += 1
    else:
        print(i)

output_y = [[0] * 10 for i in range(0, 10)]
for p in range(0, len(pertubated_label)):
    correct_index = labels.index(correct_label[p])
    predicted_index = labels.index(pertubated_label[p])

    output_y[correct_index][predicted_index] += 1

graph_data = []
for y in output_y:
    total_tests = np.sum(y)
    graph_data.append([num / total_tests for num in y])

graph_data = np.array(graph_data)
fig = plt.figure()

plt.imshow(graph_data, cmap='gist_earth', interpolation='nearest')
plt.xticks(range(0, 10), labels=labels)
plt.yticks(range(0, 10), labels=labels)
plt.colorbar()
# plt.show()
fig.savefig("graphs/pertubated_part_two.png")

index = -1
folder = "part_2_pertubations/"


for batch in range(len(pertubation_results)):
    for p in range(len(pertubation_results[batch])):
        index += 1
        if (correct_pertubated_dict[correct_label[index]][0] == pertubated_label[index]):
            print("Saving: " + str(index) +  " / " + str(len(pertubated_label)))
            sub_folder = correct_label[index] + "_" + pertubated_label[index] + "/"
            save_file = correct_label[index] + "_" + pertubated_label[index] + "_" + str(index) + ".wav"
            
            with wave.open(folder + sub_folder + save_file, 'wb') as wave_file:
                wave_file.setnchannels(1)  # mono audio
                wave_file.setsampwidth(2)  # 16-bit audio
                wave_file.setframerate(16000)  # sampling rate
                wave_file.setnframes(pertubation_results[batch][p][0].shape[0])  # number of frames
                wave_file.writeframes((pertubation_results[batch][p].cpu().numpy()[0] * 32767).astype(np.int16).tobytes())  # write audio data

        # plt.plot(pertubation_results[batch][p].cpu().numpy()[0])
        # plt.show()
        # print(audioModel.predict(pertubation_results[batch][p], attack_model, device, transform, importDataset.index_to_label, perform_transform=False))
        # plt.plot(np.array(pertubation_results[batch][p].cpu()[0]))
        # plt.show()
        # exit(0)


print("Total misclassifications:")
for key in correct_pertubated_dict:
    print(key + ": " + str(correct_pertubated_dict[key][1]) + " / " + str(correct_pertubated_dict[key][2]))