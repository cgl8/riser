import torch
from torch import nn
from torch.utils.data import DataLoader
from nets.cnn import ConvNet
from nets.resnet import ResNet
from nets.tcn import TCN
from nets.tcn_bot import TCNBot
from data import SignalDataset
from attrdict import AttrDict
import yaml

def build_loader(data_dir, batch_size, shuffle):
    dataset = SignalDataset(f"{data_dir}/positive.pt", f"{data_dir}/negative.pt")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_config(filepath):
    with open(filepath) as config_file:
        return AttrDict(yaml.load(config_file, Loader=yaml.Loader))

data_dir = "/g/data/bp00/jay/ppnpy/tensors"
config_file = "/g/data/bp00/jay/riser/riser/model/mRNA_config_R9.4.1.yaml"
model_dir = "/g/data/bp00/jay/initial_model/initial_model_0_best_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(f"Using {device} device")

config = get_config(config_file)
if config.model == 'tcn':
    model = TCN(config.tcn).to(device)
elif config.model == 'resnet':
    model = ResNet(config.resnet).to(device)
elif config.model == 'cnn':
    model = ConvNet(config.cnn).to(device)
elif config.model == 'tcn-bot':
    model = TCNBot(config.tcnbot).to(device)
else:
    print(f"{config.model} model is not supported - typo in config?")


dataloader = build_loader(f"{data_dir}/4s/test", config.batch_size, True) 

model.load_state_dict(torch.load(model_dir))


n_samples = len(dataloader.dataset)
n_batches = len(dataloader)
model.eval() #Disables dropout, etc. for model evaluation
total_loss, n_correct = 0, 0
true_pos, false_pos, true_neg, false_neg = 0,0,0,0
with torch.no_grad(): #Reduces memory usage since gradients are not calculated for backprop
    for X, y in dataloader:
        # Move batch to GPU
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        pred_values = pred.argmax(1)
        for i in range(len(y)):
            if pred_values[i]:
                if y[i]: true_pos += 1
                else: false_pos += 1
            if not pred_values[i]:
                if y[i]: false_neg += 1
                else: true_neg += 1

        n_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

# Compute average loss and accuracy
acc = n_correct / n_samples * 100
print(f"Total number of samples: {n_samples}\n")
print(f"True positives: {true_pos}\nFalse positives: {false_pos}\nTrue negatives: {true_neg}\nFalse negatives: {false_neg}")
print(f"Number correct: {n_correct}")
print(f"Test set: \n Accuracy: {acc:>0.1f}%\n")

'''
True positives: 52977
False positives: 349
True negatives: 52965
False negatives: 337
Number correct: 105942.0
Test set: 
 Accuracy: 99.4%
'''