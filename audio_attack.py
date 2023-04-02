import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import IPython.display as ipd
from torchaudio.datasets import SPEECHCOMMANDS
import os
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

 
class AdjustLength(object):
    def __init__(self, audio_length):
        super().__init__()
        self.audio_length = audio_length

    def __call__(self, sample):
        waveform = sample
        if waveform.size()[1] > self.audio_length:
            waveform = waveform[:, 0: self.audio_length]
        else:
            pad_len = self.audio_length - waveform.size()[1]
            #print(pad_len)
            pad_op = torch.nn.ZeroPad2d([0, pad_len, 0, 0])
            waveform = pad_op(waveform)
        sample = waveform
        return sample
    
    
class SubsetSC(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, file_dir="/home/zhuohang/data/", subset: str = None):
        super().__init__(file_dir, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


class SpeechCommandsDataset(Dataset):
    """Speech Command Dataset."""

    def __init__(self, file_dir='/home/zhuohang/data/', train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if train:
            self.data = SubsetSC(file_dir, subset="training")
        else:
            self.data = SubsetSC(file_dir, subset="testing")
        self.transform = transform
        self.labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
        self.targets = np.array([self.label_to_index(d[2]) for d in self.data]).astype(np.int16)
        self.speakers = np.unique([d[3] for d in self.data]).tolist()
        self.speaker_ids = np.array([self.speaker_to_index(d[3]) for d in self.data]).astype(np.int16)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        waveform, sample_rate, label, speaker_id, utterance_number = self.data[idx]
        data = waveform
        target = self.targets[idx]

        if self.transform is not None:
            data = self.transform(data)

        return data, torch.tensor(target).type(torch.LongTensor)

    def label_to_index(self, word):
        # Return the position of the word in labels
        return self.labels.index(word)

    def speaker_to_index(self, word):
        # Return the position of the word in labels
        return self.speakers.index(word)

# Create training and testing split of the data. We do not use validation in this tutorial.
trans_speech = transforms.Compose([AdjustLength(16000), torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)])
train_set = SpeechCommandsDataset('/data/', train=True, transform=trans_speech)
test_set = SpeechCommandsDataset('/data/', train=False, transform=trans_speech)   
test_set = SpeechCommandsDataset('/data/', train=False, transform=trans_speech)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=10, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        # return F.log_softmax(x, dim=2)
        return x


def _attack(model, xs, ys, eps=0.01, alpha=0.001, iters=20, targeted=False):
    xs = xs.to(device)
    ys = ys.to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_xs = xs.data
        
    for i in range(iters):
        xs.requires_grad = True
        outputs = model(xs)
    
        model.zero_grad()
        cost = loss(outputs.squeeze(), ys).to(device)
        cost.backward()
        adv_xs = xs - alpha*xs.grad.sign()
        
        eta = torch.clamp(adv_xs - ori_xs, min=-eps, max=eps)
        xs = torch.clamp(ori_xs + eta, min=-1, max=1).detach_()
            
    return xs

waveform, label = train_set[0]
data, target = next(iter(test_loader))
data, target = data.to(device), target.to(device)

model = M5(n_input=waveform.shape[0], n_output=10)
model.load_state_dict(torch.load('./models/saved_model.pth'))
model = model.to(device)


adv_data = _attack(model, data, target, eps=EPS, alpha=ALPHA, iters=ITERS, targeted=CT)