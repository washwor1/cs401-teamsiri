import torch
import torchaudio
import torch.optim as optim
from tqdm import tqdm

import audioModel
import importDataset
import matplotlib.pyplot as plt
import IPython.display as ipd

sample_rate = 16000 #All wave files have this rate
new_sample_rate = 8000 #Allows for faster training of model
batch_size = 256
log_interval = 20
n_epoch = 1


#Cuda is for GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

#get training and testing set
train_set = importDataset.SubsetSC("training")
test_set = importDataset.SubsetSC("testing")

#load labels --> I already generated this and saved it to a file
labels = []
with open("labels.txt", "r") as f:
    for line in f:
        labels.append(str(line[:-1]))

#get transform function
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transform = transform.to(device)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_loader = importDataset.getTrainLoader(train_set, batch_size, True, num_workers, pin_memory)
test_loader = importDataset.getTestLoader(test_set, batch_size, False, False, num_workers, pin_memory)


# n = count_parameters(model)


model = audioModel.M5()
model.to(device)

optimizer = audioModel.setOptimizer(model, learn_rate = 0.01, weight_decay=0.0001)
# reduce the learning after 20 epochs by a factor of 10
scheduler = audioModel.setScheduler(optimizer, step_size=20, gamma=0.1)




# losses = []
# The transform needs to live on the same device as the model and the data.

pbar_update = 1 / (len(train_loader) + len(test_loader))
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        audioModel.train(model, epoch, log_interval, train_loader, device, transform, optimizer, pbar, pbar_update)
        audioModel.test(model, epoch, test_loader, device, transform, pbar, pbar_update)
        scheduler.step()

waveform, sample_rate, utterance, *_ = train_set[50]
ipd.Audio(waveform.numpy(), rate=sample_rate)

audioModel.predict(waveform, model, device, transform, importDataset.index_to_label)
# Let's plot the training loss versus the number of iteration.
# plt.plot(losses);
# plt.title("training loss");