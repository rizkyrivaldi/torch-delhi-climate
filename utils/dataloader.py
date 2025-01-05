"""
Dataloader for torch batch training
"""

# Import generic library
import pandas as pd

# Import torch dataset library
import torch
from torch.utils.data import Dataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WindowLoaderSingleColumn(Dataset):
    """
    This only loads one column at a time, i.e:
    Load only for meantemp, humidity, wind_speed, or meanpressure
    """
    def __init__(self, dataset_path: str, column_name: str, sequence_length: int):
        # Super Init
        super().__init__()

        # Pass the parameter
        self.dataset_path = dataset_path
        self.column_name = column_name
        self.sequence_length = sequence_length

        # Set the minmax value
        """
        Config for minimum and maximum value of each column, format:
        (min_value, max_value)
        """
        self.minmax_value = {
            "meantemp" : (0.0, 100.0),
            "humidity" : (0.0, 100.0),
            "wind_speed" : (0.0, 50.0),
            "meanpressure" : (-10000.0 , 10000.0)
        }

        # Get the data with pandas
        self.df = pd.read_csv(self.dataset_path, header = 0)

        # Check if the column name is valid
        if not column_name in self.df.columns.tolist():
            raise ValueError("The column name is not available in the dataset")

        # Get the data from the selected column
        self.data_list = self.df[self.column_name].tolist()

    def __normalize(self, value):
        return (value - self.minmax_value[self.column_name][0]) / (self.minmax_value[self.column_name][1] - self.minmax_value[self.column_name][0])

    def __len__(self):
        return (len(self.df) - (self.sequence_length - 1))

    def __getitem__(self, idx):
        # Get the data sequence (idx)
        sequence_input = self.__normalize(torch.Tensor(self.data_list[idx : idx + self.sequence_length])).unsqueeze(0).transpose(1, 0).to(device)
        sequence_output = torch.Tensor(self.data_list[idx + self.sequence_length : idx + self.sequence_length + 1]).unsqueeze(0).to(device)

        return sequence_input, sequence_output