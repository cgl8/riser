import random
import sys
import time
# module load tensorflow/2.6.0
from pytorch_lightning.utilities import CombinedLoader
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from nets.cnn import ConvNet
from nets.resnet import ResNet
from nets.tcn import TCN
from nets.tcn_bot import TCNBot
from data import SignalDataset
#from utilities import get_config
from attrdict import AttrDict
import yaml


def get_config(filepath):
    with open(filepath) as config_file:
        return AttrDict(yaml.load(config_file, Loader=yaml.Loader))
'''
def count_batches_in_combined_loader(combined_loader):
    n_batches = 0
    for loader in combined_loader.flattened:
        n_batches += len(loader)
    return n_batches

def count_samples_in_combined_loader(combined_loader):
    n_samples = 0
    for loader in combined_loader.flattened:
        n_samples += len(loader.dataset)
    return n_samples
'''
def train(dataloader, model, loss_fn, optimizer, device, writer, epoch, log_freq=100):
    model.train()

    # Compute number of batches and total number of training instances
    n_samples = len(dataloader.dataset)
    print(f"Total size of training set: {n_samples}")
    n_batches = len(dataloader)
    print(f"Number of batches in training set: {n_batches}")

    # Training
    total_loss = 0.0
    batch_n = 0
    for X, y in dataloader:
        # Move batch to GPU
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if batch_n != 0 and batch_n % log_freq == 0:
            sample = batch_n * len(X)
            global_step = epoch * n_samples + sample
            avg_loss = total_loss / batch_n
            print(f"loss: {avg_loss:>7f} [{sample:>5d}/{n_samples:>5d}]")
            writer.add_scalar('training loss', avg_loss, global_step)

        # Increase batch counter
        batch_n += 1

    avg_loss = total_loss / n_batches
    return avg_loss


def validate(dataloader, model, loss_fn, device):
    n_samples = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()
    total_loss, n_correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # Move batch to GPU
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            n_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Compute average loss and accuracy
    avg_loss = total_loss / n_batches
    acc = n_correct / n_samples * 100
    print(f"Validation set: \n Accuracy: {acc:>0.1f}%, Avg loss: {avg_loss:>8f} \n")

    return avg_loss, acc


def write_scalars(writer, metrics, epoch):
    for metric, value in metrics.items():
        writer.add_scalar(metric, value, epoch)


def build_loader(data_dir, batch_size, shuffle):
    dataset = SignalDataset(f"{data_dir}/positive.pt", f"{data_dir}/negative.pt")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():

    # CL args

    exp_dir = sys.argv[1]
    data_dir = sys.argv[2]
    checkpt = sys.argv[3] if sys.argv[3] != "None" else None
    config_file = sys.argv[4]
    start_epoch = int(sys.argv[5])

    print(f"Experiment dir: {exp_dir}")
    print(f"Data dir: {data_dir}")
    print(f"Checkpoint: {checkpt}")
    print(f"Config file: {config_file}")

    # Load config

    config = get_config(config_file)

    # Determine experiment ID

    exp_id = exp_dir.split('/')[-1]

    # Create data loaders

    print("Creating data loaders...")
    #train_2s_loader = build_loader(f"{data_dir}/2s/train", config.batch_size, True)
    #train_3s_loader = build_loader(f"{data_dir}/3s/train", config.batch_size, True)
    train_4s_loader = build_loader(f"{data_dir}/4s/train", config.batch_size, True)

    #val_2s_loader = build_loader(f"{data_dir}/2s/val", config.batch_size, False)
    #val_3s_loader = build_loader(f"{data_dir}/3s/val", config.batch_size, False)
    val_4s_loader = build_loader(f"{data_dir}/4s/val", config.batch_size, False)

    # Get device for training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using {device} device")

    # Define model

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

    # Load model weights if restoring a checkpoint

    if checkpt is not None:
        model.load_state_dict(torch.load(f"{exp_dir}/{checkpt}"))
        assert start_epoch > 0
    else:
        assert start_epoch == 0
    summary(model)

    # Define loss function & optimiser

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Set up TensorBoard

    writer = SummaryWriter()

    # Train

    best_acc = 0
    best_epoch = 0
    for t in range(start_epoch, config.n_epochs):
        print(f"Epoch {t}\n-------------------------------")
        
        # Training
        start_train_t = time.time()
        train_loss = train(train_4s_loader, model, loss_fn, optimizer, device, writer, t)
        end_train_t = time.time()

        # Validation
        start_val_t = end_train_t
        val_loss, val_acc = validate(val_4s_loader, model, loss_fn, device)
        end_val_t = time.time()

        # Compute walltime taken for training and validation loops
        train_t = end_train_t - start_train_t
        val_t = end_val_t - start_val_t

        # Update TensorBoard
        metrics = {'train_loss': train_loss,
                   'val_loss': val_loss,
                   'val_acc': val_acc,
                   'train_t': train_t,
                   'val_t': val_t,
                   'train - val loss': train_loss - val_loss}
        write_scalars(writer, metrics, t)
        
        # Save model if it is the best so far
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = t
            torch.save(model.state_dict(), f"{exp_dir}/{exp_id}_{start_epoch}_best_model.pth")
            print(f"Saved best model at epoch {t} with val accuracy {best_acc}.")

        # Always save latest model in case training is interrupted
        torch.save(model.state_dict(), f"{exp_dir}/{exp_id}_latest_model.pth")
        print(f"Saved latest model at epoch {t} with val accuracy {val_acc}.")

    print(f"Best model with validation accuracy {best_acc} saved at epoch {best_epoch}.")

    print("Training complete.")



if __name__ == "__main__":
    main()
 