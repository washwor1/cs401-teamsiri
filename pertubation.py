
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
        return x
    

def attack(model, device, xs, target = None, eps=0.01, alpha=0.001, iters=20, targeted=False):
    
    xs = xs.to(device)
    target = target.to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_xs = xs.data
    
    if target == None and targeted == True:
        print("In pertubation.attack(): No target specified, but targeted is True")

    
    for i in range(iters):
        xs.requires_grad = True
        outputs = model(xs)
    
        model.zero_grad()
        cost = loss(outputs.squeeze(), target).to(device)
        cost.backward()
        adv_xs = xs - alpha*xs.grad.sign()
        
        eta = torch.clamp(adv_xs - ori_xs, min=-eps, max=eps)
        xs = torch.clamp(ori_xs + eta, min=-1, max=1).detach_()
        #print("Iteration: " + str(i))
            
    return xs   

