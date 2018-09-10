from Python.data import WiLIDataset, get_data_fields
import torch

use_cuda = True if torch.cuda.is_available() else False


def train(**kwargs):

    training_data = WiLIDataset("../Data/x_train.txt", "../Data/y_train.txt", get_data_fields())
    testing_data = WiLIDataset("../Data/x_test.txt", "../Data/y_test.txt", get_data_fields())
