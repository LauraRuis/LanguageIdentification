import io
import os
import re

from torchtext.data import Field, NestedField
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from typing import Iterable

from LanguageIdentifier.utils import PAD_TOKEN, START_TOKEN, END_TOKEN


def get_data_fields() -> dict:
    """"
    Creates torchtext fields for the I/O pipeline.
    """
    language = Field(
        batch_first=True, init_token=None, eos_token=None, pad_token=None, unk_token=None)
    characters = Field(include_lengths=True, batch_first=True, init_token=None,
                       eos_token=END_TOKEN, pad_token=PAD_TOKEN)
    nesting_field = Field(tokenize=list, pad_token=PAD_TOKEN, batch_first=True,
                          init_token=START_TOKEN, eos_token=END_TOKEN)
    
    paragraph = NestedField(nesting_field, pad_token=PAD_TOKEN, include_lengths=True)

    fields = {
        'paragraph':   ('paragraph', paragraph),
        'characters': ('characters', characters),
        'language':    ('language', language)
    }

    return fields


def empty_example() -> dict:
    ex = {
        'id':         [],
        'paragraph':  [],
        'language':   [],
        'characters': []
    }
    return ex


def data_reader(x_file: Iterable, y_file: Iterable, train: bool, split_sentences : bool) -> dict:
    """
    Return examples as a dictionary.
    """

    example = empty_example()

    for x, y in zip(x_file, y_file):

        x = x.strip()
        y = y.strip()

        examples = []

        if split_sentences:
            splitted_sentences = x.split(".")
        else:
            splitted_sentences = [x]

        for x in splitted_sentences:
            if len(x) == 0: continue
            example = empty_example()
            # replace all numbers with 0
            x = re.sub('[0-9]+', '0', x)
            paragraph = x.split()
            language = y

            if train:
                characters = list(x)[:250]
            else:
                characters = list(x)

            example['paragraph'] = [list(word.lower()) for word in paragraph]
            example['language'] = language
            example['characters'] = characters

            examples.append(example)
        yield examples

    # possible last sentence without newline after
    if len(example['paragraph']) > 0:
        yield [example]


class WiLIDataset(Dataset):

    @staticmethod
    def sort_key(example):
        return len(example.paragraph)

    def __init__(self, paragraph_path: str, label_path: str, fields: dict, split_sentences: bool, **kwargs):
        """
        Create a WiLIDataset given a path two the raw text and to the labels and field dict.
        """

        with io.open(os.path.expanduser(paragraph_path), encoding="utf8") as f_par, \
                io.open(os.path.expanduser(label_path), encoding="utf8") as f_lab:

            train = False
            if "train" in paragraph_path:
                train = True

            examples = []
            for d in data_reader(f_par, f_lab, train, split_sentences):
                for sentence in d:
                    examples.extend([Example.fromdict(sentence, fields)])

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(WiLIDataset, self).__init__(examples, fields, **kwargs)


def load_data(training_text: str, training_labels: str, testing_text: str, testing_labels: str, **kwargs) -> (WiLIDataset, WiLIDataset):

    # load training and testing data
    fields = get_data_fields()
    _paragraph = fields["paragraph"][-1]
    _language = fields["language"][-1]
    _characters = fields['characters'][-1]

    training_data = WiLIDataset(training_text, training_labels, fields, True)  # TODO: validation split
    testing_data = WiLIDataset(testing_text, testing_labels, fields, False)

    # TODO: add <unk>
    # build vocabularies
    _paragraph.build_vocab(training_data, min_freq=10)  # TODO: make min_freq parameter
    _language.build_vocab(training_data)
    _characters.build_vocab(training_data, min_freq=10)  # TODO: fix for enormous char vocab size
    return training_data, testing_data


if __name__ == "__main__":
    WiLIDataset("../Data/x_train.txt", "../Data/y_train.txt", get_data_fields(), True)
