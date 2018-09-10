from torchtext.data import Example
PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'


def print_example(example: Example):
    print()
    print("Language: ", example.language[0])
    print("Paragraph: ", (" ".join(example.paragraph)))
