import torch
from torchtext.data import Iterator

from LanguageIdentifier.data import WiLIDataset, get_data_fields
from LanguageIdentifier.utils import print_example

use_cuda = True if torch.cuda.is_available() else False
device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')


def train(training_text, training_labels, testing_text, testing_labels,**kwargs):

    training_data = WiLIDataset(training_text, training_labels, get_data_fields())
    testing_data = WiLIDataset(testing_text, testing_labels, get_data_fields())

    train_iter = Iterator(training_data, 100, train=True,
                          sort_within_batch=True, device=device)

    print_example(training_data[0])
