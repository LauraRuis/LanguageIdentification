import io
import os

from torchtext.data import Field
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from typing import Iterable

from LanguageIdentifier.utils import PAD_TOKEN


def get_data_fields() -> dict:
    """"
    Creates torchtext fields for the I/O pipeline.
    """

    paragraph = Field(
        include_lengths=True, batch_first=True,
        init_token=None, eos_token=None, pad_token=PAD_TOKEN)
    language = Field(
        include_lengths=True, batch_first=True,
        init_token=None, eos_token=None, pad_token=PAD_TOKEN)

    fields = {
        'paragraph':   ('paragraph', paragraph),
        'language':    ('language', language)
    }

    return fields


def empty_example() -> dict:
    ex = {
        'id':         [],
        'paragraph':  [],
        'language':   []
    }
    return ex


def data_reader(x_file: Iterable, y_file: Iterable) -> dict:
    """
    Return examples as a dictionary.
    """

    example = empty_example()

    for x, y in zip(x_file, y_file):
        x = x.strip()
        y = y.strip()

        example = empty_example()

        paragraph = x.split()
        language = y

        example['paragraph'] = paragraph
        example['language'] = language

        yield example

    # possible last sentence without newline after
    if len(example['paragraph']) > 0:
        yield example


class WiLIDataset(Dataset):

    @staticmethod
    def sort_key(example):
        return len(example.paragraph)

    def __init__(self, paragraph_path: str, label_path: str, fields: dict, **kwargs):
        """
        Create a WiLIDataset given a path two the raw text and to the labels and field dict.
        """

        with io.open(os.path.expanduser(paragraph_path), encoding="utf8") as f_par, \
                io.open(os.path.expanduser(label_path), encoding="utf8") as f_lab:
            examples = [Example.fromdict(d, fields) for d in data_reader(f_par, f_lab)]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(WiLIDataset, self).__init__(examples, fields, **kwargs)


if __name__ == "__main__":
    WiLIDataset("../Data/x_train.txt", "../Data/y_train.txt", get_data_fields())
