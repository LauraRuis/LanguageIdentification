#!/usr/bin/env python3

import argparse
import torch
from torchtext.data import Iterator
import os
import math

from CodeSwitching.train import train
from CodeSwitching.data import load_data
from CodeSwitching.model import GRUIdentifier, CharCNN
from CodeSwitching.utils import PAD_TOKEN

def main():
    # torch.backends.cudnn.enabled=False
    ap = argparse.ArgumentParser(description="a Language Identification model")
    ap.add_argument('--mode', choices=['train', 'predict'], default='train')
    ap.add_argument('--output_dir', type=str, default='output')

    # data arguments
    ap.add_argument('--training_text', type=str, default='Data/CodeSwitching/trn_sentences.txt')
    ap.add_argument('--training_labels', type=str, default='Data/CodeSwitching/trn_lang_labels.txt')
    ap.add_argument('--training_switch', type=str, default='Data/CodeSwitching/trn_switch_labels.txt')
    ap.add_argument('--validation_text', type=str, default='Data/CodeSwitching/dev_sentences.txt')
    ap.add_argument('--validation_labels', type=str, default='Data/CodeSwitching/dev_lang_labels.txt')
    ap.add_argument('--validation_switch', type=str, default='Data/CodeSwitching/dev_switch_labels.txt')
    ap.add_argument('--testing_text', type=str, default='Data/CodeSwitching/tst_sentences.txt')
    ap.add_argument('--testing_labels', type=str, default='Data/CodeSwitching/tst_lang_labels.txt')
    ap.add_argument('--testing_switch', type=str, default='Data/CodeSwitching/tst_switch_labels.txt')
    ap.add_argument('--level', type=str, choices=['word', 'char'], default='char')

    # general model parameters
    ap.add_argument('--model_type', type=str, default='recurrent')
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=10)

    # logging parameters
    ap.add_argument('--eval_frequency', type=int, default=100)
    ap.add_argument('--log_frequency', type=int, default=100)
    ap.add_argument('--resume_from_file', type=str, default="")

    # Recurrent model settings
    ap.add_argument('--embedding_dim', type=int, default=100)
    ap.add_argument('--hidden_dim', type=int, default=100)
    ap.add_argument('--bidirectional', action='store_true')

    cfg = vars(ap.parse_args())

    print("Parameters:")
    for k, v in cfg.items():
        print("  %12s : %s" % (k, v))
    print()

    # Check for GPU
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')

    # Load datasets and create iterators to use while training / testing
    training_data, validation_data, testing_data = load_data(**cfg)

    print("Data loaded.")
    if cfg['mode'] == 'train':
        training_iterator = Iterator(training_data, cfg['batch_size'], train=True,
                                     sort_within_batch=True, device=device, repeat=False)
        validation_iterator = Iterator(validation_data, cfg['batch_size'], train=False, sort_within_batch=True,
                                       device=-1, repeat=False)
    testing_iterator = Iterator(testing_data, cfg['batch_size'], train=False,
                                sort_within_batch=True, device=-1, repeat=False)

    print("Loaded %d training samples" % len(training_data))
    print("Loaded %d validation samples" % len(validation_data))
    print("Loaded %d test samples" % len(testing_data))
    print()

    if cfg['mode'] == 'train':

        # Calculate args needed for recurrent model, move these lines if used
        # by other models / pieces of code

        if cfg['level'] == 'char':
            vocab_size = len(training_data.fields['characters'].vocab)
        else:
            vocab_size = len(training_data.fields['paragraph'].vocab)
        n_classes = len(training_data.fields['language_per_word'].vocab)

        # Initialise a new model
        if cfg['model_type'] == 'recurrent':
            model = GRUIdentifier(vocab_size, n_classes, **cfg)
        elif cfg['model_type'] == 'character_cnn':
            padding_idx = training_data.fields['characters'].vocab.stoi[PAD_TOKEN]
            model = CharCNN(vocab_size, padding_idx,
                            emb_dim=cfg["embedding_dim"], num_filters=30, window_size=3, dropout_p=0.33,
                            n_classes=n_classes)
        elif cfg['model_type'] == 'cnn_rnn':
            training_data.fields['paragraph']
            char_vocab_size = len(training_data.fields['characters'].vocab)
            d = round(math.log(abs(char_vocab_size)))
            model = CNNRNN(char_vocab_size, d, vocab_size, n_classes, num_filters=1, kernel_size=3, n1=1, n2=1)

        else:
            raise NotImplementedError()

        print("Vocab. size word: ", len(training_data.fields['paragraph'].vocab))
        print("First 10 words: ", " ".join(training_data.fields['paragraph'].vocab.itos[:10]))
        print("Vocab. size chars: ", len(training_data.fields['characters'].vocab))
        print("First 10 chars: ", " ".join(training_data.fields['characters'].vocab.itos[:10]))
        print("Number of languages: ", n_classes)
        print()

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

        if cfg["resume_from_file"]:

           if os.path.isfile(cfg["resume_from_file"]):

               file = cfg["resume_from_file"]
               print("Loading model from file '{}'".format(file))
               resume_state = torch.load(file)
               model.load_state_dict(resume_state['state_dict'])
               optimizer.load_state_dict(resume_state['optimizer'])
               print("Loaded from file '{}' (epoch {})"
                     .format(file, resume_state['epoch']))
           else:
               resume_state = None
               print("=> no checkpoint found at '{}'".format(cfg["resume_from_file"]))
        else:
            resume_state = None

        if use_cuda: model.cuda()

        train(model=model, optimizer=optimizer,
              training_data=training_iterator, validation_data=validation_iterator, testing_data=testing_iterator,
              optimizer=optimizer,
              resume_state=resume_state, **cfg)

    elif cfg['mode'] == 'test':  # Let's separate test from inference mode
        raise NotImplementedError()
        # Load model
        # Run test() from test.py
    elif cfg['mode'] == 'predict':  # Let's separate test from inference mode
        raise NotImplementedError()


if __name__ == '__main__':
    main()
