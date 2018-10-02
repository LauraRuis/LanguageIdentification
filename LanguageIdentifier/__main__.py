#!/usr/bin/env python3

import argparse
import torch
from torchtext.data import Iterator
import os
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import math

from LanguageIdentifier.train import train
from LanguageIdentifier.data import load_data
from LanguageIdentifier.model import GRUIdentifier, CharCNN, SmallCNN, CNNRNN
from LanguageIdentifier.utils import PAD_TOKEN


def main():
    # torch.backends.cudnn.enabled=False
    ap = argparse.ArgumentParser(description="a Language Identification model")
    ap.add_argument('--mode', choices=['train', 'predict'], default='train')
    ap.add_argument('--output_dir', type=str, default='output')

    # data arguments
    ap.add_argument('--training_text', type=str, default='Data/x_train_split.txt')
    ap.add_argument('--training_labels', type=str, default='Data/y_train_split.txt')
    ap.add_argument('--validation_text', type=str, default='Data/x_valid_split.txt')
    ap.add_argument('--validation_labels', type=str, default='Data/y_valid_split.txt')
    ap.add_argument('--testing_text', type=str, default='Data/x_test_split.txt')
    ap.add_argument('--testing_labels', type=str, default='Data/y_test_split.txt')

    # data parameters
    ap.add_argument('--max_chars', type=int, default=250)
    ap.add_argument('--split_paragraphs', action='store_true', default=False)
    ap.add_argument('--fix_lengths', action='store_true', default=False)

    # general model parameters
    ap.add_argument('--model_type', type=str, default='recurrent')
    ap.add_argument('--level', type=str, default='char')
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--optimizer', type=str, default='adam')

    # logging parameters
    ap.add_argument('--eval_frequency', type=int, default=100)
    ap.add_argument('--log_frequency', type=int, default=100)
    ap.add_argument('--resume_from_file', type=str, default="")

    # Recurrent model settings
    ap.add_argument('--embedding_dim', type=int, default=100)
    ap.add_argument('--hidden_dim', type=int, default=100)
    ap.add_argument('--bidirectional', action='store_true')

    cfg = vars(ap.parse_args())

    if cfg["model_type"] == "large_cnn":
        assert cfg["fix_lengths"], "Please set flage fix_lengths to true when using large cnn " \
                                      "(fixed length input necessary)"
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
                                       device=device, repeat=False)
    testing_iterator = Iterator(testing_data, cfg['batch_size'], train=False,
                                sort_within_batch=True, device=device, repeat=False)

    print("Loaded %d training samples" % len(training_data))
    print("Loaded %d validation samples" % len(validation_data))
    print("Loaded %d test samples" % len(testing_data))
    print()

    if cfg['mode'] == 'train':

        # Calculate args needed for recurrent model, move these lines if used
        # by other models / pieces of code
        char_vocab_size = len(training_data.fields['characters'].vocab)
        n_classes = len(training_data.fields['language'].vocab)

        # Initialise a new model
        if cfg['model_type'] == 'recurrent':

            model = GRUIdentifier(char_vocab_size, n_classes, **cfg)
        elif cfg['model_type'] == 'small_cnn':
            padding_idx = training_data.fields['characters'].vocab.stoi[PAD_TOKEN]
            model = SmallCNN(char_vocab_size, padding_idx, emb_dim=cfg["embedding_dim"], dropout_p=0.33, num_filters=30,
                             window_size=3, n_classes=n_classes)
        elif cfg['model_type'] == 'large_cnn':
            padding_idx = training_data.fields['characters'].vocab.stoi[PAD_TOKEN]
            model = CharCNN(char_vocab_size, padding_idx, emb_dim=cfg["embedding_dim"],
                            dropout_p=0.5, n_classes=n_classes, length=cfg['max_chars'],)
        elif cfg['model_type'] == 'cnn_rnn':
            char_vocab_size = len(training_data.fields['paragraph'].vocab)
            d = round(math.log(abs(char_vocab_size)))
            model = CNNRNN(char_vocab_size, d, n_classes, num_filters=50, kernel_size=3, n1=1,
                           vocab=training_data.fields['paragraph'].vocab.itos)
        else:
            raise NotImplementedError()

        print("Vocab. size word: ", len(training_data.fields['paragraph'].vocab))
        print("First 10 words: ", " ".join(training_data.fields['paragraph'].vocab.itos[:10]))
        print("Vocab. size chars: ", len(training_data.fields['characters'].vocab))
        print("First 10 chars: ", " ".join(training_data.fields['characters'].vocab.itos[:10]))
        print("Number of languages: ", n_classes)
        print()

        if cfg["optimizer"] == "adam":
            par_optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
            scheduler = None
        elif cfg["optimizer"] == "sgd":
            par_optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"], momentum=0.9)
            scheduler = LambdaLR(par_optimizer, lr_lambda=lambda t: 0.8**(t / 3))
        else:
            raise NotImplementedError()

        print("Trainable Parameters")
        n_params = sum([np.prod(par.size()) for par in model.parameters() if par.requires_grad])
        print("Number of parameters: {}".format(n_params))
        for name, par in model.named_parameters():
            if par.requires_grad:
                print("{} : {}".format(name, list(par.size())))
        print()

        if cfg["resume_from_file"]:

           if os.path.isfile(cfg["resume_from_file"]):

               file = cfg["resume_from_file"]
               print("Loading model from file '{}'".format(file))
               resume_state = torch.load(file)
               model.load_state_dict(resume_state['state_dict'])
               par_optimizer.load_state_dict(resume_state['optimizer'])
               print("Loaded from file '{}' (epoch {})"
                     .format(file, resume_state['epoch']))
           else:
               resume_state = None
               print("=> no checkpoint found at '{}'".format(cfg["resume_from_file"]))
        else:
            resume_state = None

        if use_cuda: model.cuda()

        train(model=model,
              training_data=training_iterator, validation_data=validation_iterator, testing_data=testing_iterator,
              par_optimizer=par_optimizer, scheduler=scheduler,
              resume_state=resume_state, **cfg)

    elif cfg['mode'] == 'test':  # Let's separate test from inference mode
        raise NotImplementedError()
        # Load model
        # Run test() from test.py
    elif cfg['mode'] == 'predict':  # Let's separate test from inference mode
        raise NotImplementedError()


if __name__ == '__main__':
    main()
