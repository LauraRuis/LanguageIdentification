from torchtext.data import Example
from torchtext.data import Field
from typing import Iterable
import os

PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'


def print_example(example: Example):
    print()
    print("Language: ", example.language[0])
    print("Paragraph: ", (" ".join(example.paragraph)))


def calculate_char_freqs(data: Iterable, language_field: Field):
    languages = []
    vocab_sizes = []
    for i, language in enumerate(language_field.vocab.itos):

        char_set = set()
        for example in data:
            lang = example.language
            if not lang[0] == language:
                continue
            else:
                chars = example.characters
                for word in chars:
                    for char in word:
                        char_set.add(char)

        languages.append(str(language))
        vocab_sizes.append(len(char_set))

    vocab_sizes, languages = zip(*sorted(zip(vocab_sizes, languages)))
    info = []
    for size, language in zip(vocab_sizes, languages):
        language_info = language + "\t" + str(size) + "\n"
        info.append(language_info)

    with open("char_freqs_temp.txt", "w") as infile:
        infile.writelines(info)
    print("Wrote to: ", os.getcwd() + "/char_freqs_temp.txt")