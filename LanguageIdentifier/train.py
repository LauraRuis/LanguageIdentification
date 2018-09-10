import torch
from torchtext.data import Iterator
import os

from LanguageIdentifier.data import WiLIDataset, get_data_fields
from LanguageIdentifier.utils import print_example, calculate_char_freqs

use_cuda = True if torch.cuda.is_available() else False
device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')


def train(training_text: str, training_labels: str, testing_text: str, testing_labels: str,
          **kwargs):

    # load training and testing data
    fields = get_data_fields()
    _paragraph = fields["paragraph"][-1]
    _language = fields["language"][-1]
    _characters = fields['characters'][-1]

    training_data = WiLIDataset(training_text, training_labels, fields)
    testing_data = WiLIDataset(testing_text, testing_labels, fields)
    training_iterator = Iterator(training_data, 100, train=True,
                                 sort_within_batch=True, device=device)  # TODO: batchsize 100 to arg in __main__

    # print first training example
    print("First training example: ")
    print_example(training_data[0])

    # TODO: add <unk>
    # build vocabularies
    _paragraph.build_vocab(training_data, min_freq=1)
    _language.build_vocab(training_data)
    # _characters.build_vocab(training_data, min_freq=1000)  # TODO: fix for enormous char vocab size

    # example batch
    # batch = next(iter(training_iterator))
