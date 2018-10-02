import io
import os
import re

from torchtext.data import Field, NestedField
import torchtext.data as data
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from typing import Iterable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from LanguageIdentifier.utils import PAD_TOKEN, START_TOKEN, END_TOKEN

from nltk.tokenize import sent_tokenize


def get_data_fields(fixed_lengths: int) -> dict:
    """"
    Creates torchtext fields for the I/O pipeline.
    """
    language = Field(
        batch_first=True, init_token=None, eos_token=None, pad_token=None, unk_token=None)

    characters = Field(include_lengths=True, batch_first=True, init_token=None,
                       eos_token=END_TOKEN, pad_token=PAD_TOKEN, fix_length=fixed_lengths)

    nesting_field = Field(tokenize=list, pad_token=PAD_TOKEN, batch_first=True,
                          init_token=None, eos_token=END_TOKEN)
    paragraph = NestedField(nesting_field, pad_token=PAD_TOKEN, eos_token=END_TOKEN,
                            include_lengths=True)
    #
    # paragraph = Field(include_lengths=True, batch_first=True, init_token=None,
    #                   eos_token=END_TOKEN, pad_token=PAD_TOKEN)

    fields = {
        'characters': ('characters', characters),
        'paragraph':   ('paragraph', paragraph),
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


def data_reader(x_file: Iterable, y_file: Iterable, train: bool, split_sentences, max_chars: int) -> dict:
    """
    Return examples as a dictionary.
    """

    example = empty_example()
    spacy_tokenizer = data.get_tokenizer("spacy")  # TODO: implement with word level

    for x, y in zip(x_file, y_file):

        x = x.strip()
        y = y.strip()

        examples = []

        if split_sentences:
            splitted_sentences = sent_tokenize(x)
        else:
            splitted_sentences = [x]

        for x in splitted_sentences:
            if len(x) == 0: continue
            example = empty_example()

            # replace all numbers with 0
            x = re.sub('[0-9]+', '0', x)
            # x = spacy_tokenizer(x)
            paragraph = x.split()
            language = y

            example['paragraph'] = [word.lower() for word in paragraph]
            example['language'] = language
            example['characters'] = list(x) if not train else list(x)[:max_chars]

            examples.append(example)
        yield examples

    # possible last sentence without newline after
    if len(example['paragraph']) > 0:
        yield [example]


class WiLIDataset(Dataset):

    @staticmethod
    def sort_key(example):
        return len(example.characters)

    def __init__(self, paragraph_path: str, label_path: str, fields: dict, split_sentences: bool, train: bool,
                 max_chars: int=1000, level: str="char",
                 **kwargs):
        """
        Create a WiLIDataset given a path two the raw text and to the labels and field dict.
        """

        self.level = level

        with io.open(os.path.expanduser(paragraph_path), encoding="utf8") as f_par, \
                io.open(os.path.expanduser(label_path), encoding="utf8") as f_lab:

            examples = []
            for d in data_reader(f_par, f_lab, train, split_sentences, max_chars):
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


def load_data(training_text: str, training_labels: str, testing_text: str, testing_labels: str,
              validation_text: str, validation_labels: str, max_chars: int=1000,
              split_paragraphs: bool=False, fix_lengths: bool=False, level: str="char", **kwargs) -> (WiLIDataset, WiLIDataset):

    # load training and testing data
    if fix_lengths:
        fixed_length = max_chars
    else:
        fixed_length = None

    fields = get_data_fields(fixed_length)
    _paragraph = fields["paragraph"][-1]
    _language = fields["language"][-1]
    _characters = fields['characters'][-1]

    training_data = WiLIDataset(training_text, training_labels, fields, split_paragraphs, True, max_chars, level)
    validation_data = WiLIDataset(validation_text, validation_labels, fields, False, False, max_chars, level)
    testing_data = WiLIDataset(testing_text, testing_labels, fields, False, False, max_chars, level)

    # TODO: add <unk>
    # build vocabularies
    _paragraph.build_vocab(training_data, min_freq=10)  # TODO: make min_freq parameter
    _language.build_vocab(training_data)
    _characters.build_vocab(training_data, min_freq=10)

    return training_data, validation_data, testing_data


# if __name__ == "__main__":
    # WiLIDataset("../Data/x_train.txt", "../Data/y_train.txt", get_data_fields(), True)
