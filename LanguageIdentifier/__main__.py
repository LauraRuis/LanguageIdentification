#!/usr/bin/env python3

import argparse
import torch
from torchtext.data import Iterator

from train import train
from data import load_data
from model import GRUIdentifier


def main():
    torch.backends.cudnn.enabled=False
    ap = argparse.ArgumentParser(description="a Language Identification model")
    ap.add_argument('--mode', choices=['train', 'predict'], default='train')
    ap.add_argument('--output_dir', type=str, default='output')

    ap.add_argument('--training_text', type=str, default='Data/x_train.txt')
    ap.add_argument('--training_labels', type=str, default='Data/y_train.txt')
    ap.add_argument('--testing_text', type=str, default='Data/x_test.txt')
    ap.add_argument('--testing_labels', type=str, default='Data/y_test.txt')
    ap.add_argument('--model', type=str, default='recurrent')
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=10)

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
    training_data, testing_data = load_data(**cfg)
    if cfg['mode'] == 'train':
        training_iterator = Iterator(training_data, cfg['batch_size'], train=True,
                                     sort_within_batch=True, device=device, repeat=False)
    testing_iterator = Iterator(testing_data, cfg['batch_size'], train=True,
                                sort_within_batch=True, device=device, repeat=False)

    if cfg['mode'] == 'train':
        # Initialise a new model
        if cfg['model'] == 'recurrent':
            # Calculate args needed for recurrent model, move these lines if used
            # by other models / pieces of code
            vocab_size = len(training_data.fields['paragraph'].vocab)
            n_classes = len(training_data.fields['language'].vocab)
            model = GRUIdentifier(vocab_size, n_classes, **cfg)
        else:
            raise NotImplementedError()
        if use_cuda: model.cuda()

        train(model, training_iterator, testing_iterator, 
              cfg['learning_rate'], cfg['epochs'])

    elif cfg['mode'] == 'test': # Let's separate test from inference mode
        raise NotImplementedError()
        # Load model
        # Run test() from test.py
    elif cfg['mode'] == 'predict': # Let's separate test from inference mode
        raise NotImplementedError()


if __name__ == '__main__':
    main()
