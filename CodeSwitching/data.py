import io
import os
import re
import string

from torchtext.data import Field, NestedField
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from typing import Iterable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from CodeSwitching.utils import PAD_TOKEN, START_TOKEN, END_TOKEN

from nltk.tokenize import sent_tokenize


def get_data_fields() -> dict:
    """"
    Creates torchtext fields for the I/O pipeline.
    """
    language_per_word = Field(include_lengths=True, batch_first=True, init_token=None, 
                              eos_token=END_TOKEN, pad_token=PAD_TOKEN)
    language_per_char = Field(include_lengths=True, batch_first=True, init_token=None, 
                              eos_token=END_TOKEN, pad_token=PAD_TOKEN)
    characters = Field(include_lengths=True, batch_first=True, init_token=None,
                       eos_token=END_TOKEN, pad_token=PAD_TOKEN)
    paragraph = Field(include_lengths=True, batch_first=True, init_token=None,
                      eos_token=END_TOKEN, pad_token=PAD_TOKEN)

    fields = {
        'characters': ('characters', characters),
        'paragraph':   ('paragraph', paragraph),
        'language_per_word' : ('language_per_word', language_per_word),
        'language_per_char' : ('language_per_char', language_per_char)
    }

    return fields


def empty_example() -> dict:
    ex = {
        'id':         [],
        'paragraph':  [],
        'characters': [],
        'language_per_char':   [],
        'language_per_word':   []
    }
    return ex


def data_reader(x_file: Iterable, y_file: Iterable, switch_file : Iterable) -> dict:
    """
    Return examples as a dictionary.
    """

    example = empty_example()

    for x, y, switch in zip(x_file, y_file, switch_file):

        x = x.strip()
        y = y.strip()
        switch = int(switch.strip())

        if len(x) == 0: continue
        example = empty_example()

        # replace all numbers with 0
        x = re.sub('[0-9]+', '0', x)
        first_lang, second_lang = tuple(y.split(","))

        # Collect characters and target for character
        example['characters'] = list(x)
        example['language_per_char'] = [first_lang for char in x[:switch]] + \
                                       [second_lang for char in x[switch:]]

        # Collect targets per word, represent punctuation as words
        first_lang_text = x[:switch]
        second_lang_text = x[switch:]

        for token in string.punctuation:
            if token != "." and token != ":" and token != "!" and token != ",":
                first_lang_text = first_lang_text.replace(token, "")
                second_lang_text = second_lang_text.replace(token, "")
            else:
                first_lang_text = first_lang_text.replace(token, " {} ".format(token))
                second_lang_text = second_lang_text.replace(token, " {} ".format(token))

        first_lang_text = re.sub(' +', ' ', first_lang_text)
        second_lang_text = re.sub(' +', ' ', second_lang_text)
        paragraph = first_lang_text.split() + second_lang_text.split()

        example['paragraph'] = [word for word in paragraph]
        example['language_per_word'] = [first_lang for word in first_lang_text.split()] + \
                                       [second_lang for word in second_lang_text.split()]

        yield example

    # possible last sentence without newline after
    if len(example['paragraph']) > 0:
        yield None


class WiLIDataset(Dataset):

    def sort_key(self, example):
        if self.level == 'char':
            return len(example.characters)
        else:
            return len(example.paragraph)

    def __init__(self, paragraph_path: str, label_path: str, switch_path : str,
                 fields: dict, level : str, **kwargs):
        """
        Create a WiLIDataset given a path two the raw text and to the labels and field dict.
        """

        self.level = level

        with io.open(os.path.expanduser(paragraph_path), encoding="utf8") as f_par, \
             io.open(os.path.expanduser(label_path), encoding="utf8") as f_lang, \
                io.open(os.path.expanduser(switch_path), encoding="utf8") as f_switch:

            examples = []
            for d in data_reader(f_par, f_lang, f_switch):
                if d is None: continue
                examples.append(Example.fromdict(d, fields))

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(WiLIDataset, self).__init__(examples, fields, **kwargs)


def load_data(training_text: str, training_labels: str, training_switch : str, 
              testing_text: str, testing_labels: str, testing_switch: str,
              validation_text: str, validation_labels: str, validation_switch : str,
              level : str, **kwargs) -> (WiLIDataset, WiLIDataset, WiLIDataset):

    # load training and testing data
    fields = get_data_fields()
    _language_per_char = fields["language_per_char"][-1]
    _language_per_word = fields["language_per_word"][-1]
    _paragraph = fields["paragraph"][-1]
    _characters = fields['characters'][-1]

    training_data = WiLIDataset(training_text, training_labels, training_switch, fields, level)
    validation_data = WiLIDataset(validation_text, validation_labels, validation_switch, fields, level)
    testing_data = WiLIDataset(testing_text, testing_labels, testing_switch, fields, level)

    # TODO: add <unk>
    # build vocabularies
    _language_per_word.build_vocab(training_data)
    _language_per_char.build_vocab(training_data)
    _paragraph.build_vocab(training_data, min_freq=1)  # TODO: make min_freq parameter
    _characters.build_vocab(training_data, min_freq=1)  # TODO: fix for enormous char vocab size

    return training_data, validation_data, testing_data


if __name__ == "__main__":
    dataset = WiLIDataset("Data/CodeSwitching/trn_sentences.txt",
                         "Data/CodeSwitching/trn_lang_labels.txt",
                         "Data/CodeSwitching/trn_switch_labels.txt",
                         get_data_fields())
    print(dataset[0].characters)
