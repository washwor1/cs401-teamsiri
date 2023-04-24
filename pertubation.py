
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
    
# import torch
# import torch.nn as nn
# import torch.optim as optim

# def cw_loss(logits, target, kappa=0):
#     # Helper function to compute the Carlini & Wagner loss
#     batch_size = logits.size(0)
#     num_classes = logits.size(-1)  # Calculate the number of classes from logits tensor
    
#     target = target.to(logits.device, dtype=torch.long)  # Ensure target has the correct data type
#     target_onehot = torch.zeros(batch_size, num_classes, device=logits.device)
#     target_onehot.scatter_(1, target.view(-1, 1), 1)
    
#     real = torch.sum(target_onehot * logits, dim=1)
#     other = torch.max((1 - target_onehot) * logits, dim=1)[0]

#     loss = torch.clamp(other - real + kappa, min=0)
#     return torch.mean(loss)

# def attack(model, device, xs, target, eps=1.0, iters=100, targeted=False, learning_rate=0.01, kappa_candidates=None):

#     xs = xs.to(device)
#     target = target.to(device)
#     ori_xs = xs.clone().detach()

#     if target is None and targeted is True:
#         print("In pertubation.attack(): No target specified, but targeted is True")

#     delta = torch.zeros_like(xs, requires_grad=True)
#     optimizer = optim.Adam([delta], lr=learning_rate)

#     if kappa_candidates is None:
#         kappa_candidates = [0, 1, 5, 10, 20, 50]

#     best_adv_xs = None
#     best_adv_distance = float('inf')

#     for kappa in kappa_candidates:
#         for i in range(iters):
#             adv_xs = xs + delta
#             adv_xs = torch.clamp(adv_xs, min=-1, max=1)

#             outputs = model(adv_xs)

#             model.zero_grad()
#             cost = cw_loss(outputs, target, kappa=kappa)

#             cost.backward()
#             optimizer.step()

#             delta.data = torch.clamp(delta.data, min=-eps, max=eps)

#         adv_xs = torch.clamp(ori_xs + delta, min=-1, max=1)

#         outputs_labels = torch.argmax(outputs, dim=1).view(-1)
#         target_labels = target.view(-1)

#         # Workaround for incorrect output shape
#         if outputs_labels.shape[0] != target_labels.shape[0]:
#             num_batches = outputs_labels.shape[0] // target_labels.shape[0]
#             outputs_labels = outputs_labels.view(num_batches, -1)
#             outputs_labels = outputs_labels.to(dtype=torch.float).mean(dim=0).round().to(dtype=torch.long)

#         if targeted:
#             successful = (outputs_labels == target_labels)
#         else:
#             successful = (outputs_labels != target_labels)


#         if successful.any():
#             adv_distance = torch.norm(adv_xs - ori_xs, p=2, dim=1)
#             successful_indices = torch.nonzero(successful, as_tuple=False).squeeze()

#             for idx in successful_indices:
#                 if adv_distance[idx] < best_adv_distance:
#                     best_adv_xs = adv_xs[idx].unsqueeze(0)
#                     best_adv_distance = adv_distance[idx]

#     if best_adv_xs is None:
#         best_adv_xs = adv_xs

#     return best_adv_xs


# def cw_loss(logits, target, kappa=0):
#     # Helper function to compute the Carlini & Wagner loss
#     batch_size = logits.size(0)
#     num_classes = logits.size(-1)  # Calculate the number of classes from logits tensor
    
#     target = target.to(logits.device, dtype=torch.long)  # Ensure target has the correct data type
#     target_onehot = torch.zeros(batch_size, num_classes, device=logits.device)
#     target_onehot.scatter_(1, target.view(-1, 1), 1)
    
#     real = torch.sum(target_onehot * logits, dim=1)
#     other = torch.max((1 - target_onehot) * logits, dim=1)[0]

#     loss = torch.clamp(other - real + kappa, min=0)
#     return torch.mean(loss)





# def attack(model, device, xs, target, eps=0.01, alpha=0.01, iters=50, targeted=False, learning_rate=0.01):

#     xs = xs.to(device)
#     target = target.to(device)
#     ori_xs = xs.clone().detach()

#     if target is None and targeted is True:
#         print("In pertubation.attack(): No target specified, but targeted is True")

#     delta = torch.zeros_like(xs, requires_grad=True)

#     optimizer = optim.Adam([delta], lr=learning_rate)

#     for i in range(iters):
#         adv_xs = xs + delta
#         adv_xs = torch.clamp(adv_xs, min=-1, max=1)

#         outputs = model(adv_xs)

#         model.zero_grad()
#         cost = cw_loss(outputs, target)

#         cost.backward()
#         optimizer.step()

#         delta.data = torch.clamp(delta.data, min=-eps, max=eps)

#         print("Iteration: " + str(i))

#     adv_xs = torch.clamp(ori_xs + delta, min=-1, max=1)
#     return adv_xs



                                            # .01, .001
def attack(model, device, batch, target = None, eps=.01, alpha=0.001, iters=200, targeted=False):
    
    batch = batch.to(device)
    target = target.to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_batch = batch.data
    
    if target == None and targeted == True:
        print("In pertubation.attack(): No target specified, but targeted is True")

    
    for i in range(iters):
        batch.requires_grad = True
        outputs = model(batch)

        model.zero_grad()
        cost = loss(outputs.squeeze(), target).to(device)
        cost.backward()
        adv_batch = batch - alpha*batch.grad.sign()
        
        eta = torch.clamp(adv_batch - ori_batch, min=-eps, max=eps)
        batch = torch.clamp(ori_batch + eta, min=-1, max=1).detach_()
        print("Iteration: " + str(i))
            
    return batch   

