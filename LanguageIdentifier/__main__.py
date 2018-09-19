#!/usr/bin/env python3

import argparse
import torch
from torchtext.data import Iterator
import os

from LanguageIdentifier.train import train
from LanguageIdentifier.data import load_data
from LanguageIdentifier.model import GRUIdentifier, CharCNN
from LanguageIdentifier.utils import PAD_TOKEN


def main():
    torch.backends.cudnn.enabled=False
    ap = argparse.ArgumentParser(description="a Language Identification model")
    ap.add_argument('--mode', choices=['train', 'predict'], default='train')
    ap.add_argument('--output_dir', type=str, default='output')

    ap.add_argument('--training_text', type=str, default='Data/x_train_split.txt')
    ap.add_argument('--training_labels', type=str, default='Data/y_train_split.txt')
    ap.add_argument('--validation_text', type=str, default='Data/x_valid_split.txt')
    ap.add_argument('--validation_labels', type=str, default='Data/y_valid_split.txt')
    ap.add_argument('--testing_text', type=str, default='Data/x_test_split.txt')
    ap.add_argument('--testing_labels', type=str, default='Data/y_test_split.txt')
    ap.add_argument('--model_type', type=str, default='recurrent')
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--eval_frequency', type=int, default=100)
    ap.add_argument('--resume_from_file', type=str, default="")

    # Recurrent model settings
    ap.add_argument('--embedding_dim', type=int, default=100)
    ap.add_argument('--hidden_dim', type=int, default=100)
    ap.add_argument('--bidirectional', action='store_true')

    cfg = vars(ap.parse_args())

    print("Parameters:")
    for k, v in cfg.items():
        print("  %12s : %s" % (k, v))

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

    if cfg['mode'] == 'train':

        # Initialise a new model
        if cfg['model_type'] == 'recurrent':
            # Calculate args needed for recurrent model, move these lines if used
            # by other models / pieces of code
            vocab_size = len(training_data.fields['paragraph'].vocab)
            n_classes = len(training_data.fields['language'].vocab)
            model = GRUIdentifier(vocab_size, n_classes, **cfg)
        elif cfg['model_type'] == 'character_cnn':
            vocab_size = len(training_data.fields['characters'].vocab)
            n_classes = len(training_data.fields['language'].vocab)
            padding_idx = training_data.fields['characters'].vocab.stoi[PAD_TOKEN]
            model = CharCNN(vocab_size, padding_idx,
                            emb_dim=cfg["embedding_dim"], num_filters=30, window_size=3, dropout_p=0.33,
                            n_classes=n_classes)
        else:
            raise NotImplementedError()

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

        train(model=model,
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
