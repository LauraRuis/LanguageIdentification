PAD_TOKEN = '<pad>'


def print_example(example):
    print()
    print("Language: ", example.language[0])
    print("Paragraph: ", (" ".join(example.paragraph)))
