import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

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
        return F.log_softmax(x, dim=2)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def train(model, epoch, log_interval, train_loader, device, transform, optimizer, pbar, pbar_update):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        #losses.append(loss.item())


def test(model, epoch, test_loader, device, transform, pbar, pbar_update):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


def setOptimizer(model, learn_rate, weight_decay):
    return optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=learn_rate, weight_decay=weight_decay)

def setScheduler(optimizer, step_size, gamma):
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def predict(tensor, model, device, transform, index_to_label, perform_transform):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device)
    if(perform_transform == True):
        tensor = transform(tensor)
    #print(tensor.size())
    tensor = model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor


#validate --> Probably will get rid of later. 
def compute_loss(outputs, target):
    pass

def compute_metrics(outputs, target):
    pass

def validate(model, epoch, validation_loader, device, transform, pbar, pbar_update):

    model.eval()
    loss = None
    metrics = None

    with torch.no_grad():
        for data, target in validation_loader:
            # Move inputs and targets to device (GPU or CPU)
            inputs = inputs.to(device)
            target = target.to(device)

            data = transform(data)
            outputs = model(data)

            # Compute loss and other metrics
            loss = compute_loss(outputs, target)
            metrics = compute_metrics(outputs, target)



#code for validation stuff

"""
import torch
from torch.utils.data import DataLoader
from my_dataset import TestDataset # replace with your own dataset class

# Step 1: Load the test dataset
test_dataset = TestDataset(...)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 2: Load the trained model
model = torch.load('path/to/model.pth')

# Step 3: Set the model to evaluation mode
model.eval()

# Step 4: Run the validation loop
with torch.no_grad():
    for inputs, targets in test_loader:
        # Move inputs and targets to device (GPU or CPU)
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Pass inputs through the model to generate predictions
        outputs = model(inputs)

        # Compute loss and other metrics
        loss = compute_loss(outputs, targets)
        metrics = compute_metrics(outputs, targets)

# Step 5: Compute the validation metrics
validation_metric = compute_validation_metric(metrics)
print('Validation Metric: {}'.format(validation_metric))

"""