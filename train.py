# External Imports
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
import os

# Local Import
from utils.utils import AttrDict
from utils.dataloader import WindowLoaderSingleColumn
from model import LSTMModel

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # Get the config data
    with open('config.yaml') as f:
        # Load data to dictionary
        config_dict = yaml.safe_load(f)

        # Convert the dictionary to object
        config = AttrDict(config_dict)

    # Get the dataloader
    train_loader = DataLoader(WindowLoaderSingleColumn("dataset/DailyDelhiClimateTrain.csv", column_name = "meantemp", sequence_length = 5), batch_size = config.batch_size)
    # train_loader.__getitem__(0)

    # Get the model
    model = LSTMModel(config).to(device)

    for a, b in train_loader:
        print("BLOCKER")
        pred = model(a, (torch.randn(1, config.batch_size, config.hidden_neuron).to(device), torch.randn(1, config.batch_size, config.hidden_neuron).to(device)))
        print("Prediction")
        print(pred[0])
        print(pred[1])
        # print(f"Input data {a.shape}")
        # print(f"Output data {b.shape}")
