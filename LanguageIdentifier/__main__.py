#!/usr/bin/env python3

import argparse
from LanguageIdentifier.train import train


def main():

    ap = argparse.ArgumentParser(description="a Language Identification model")
    ap.add_argument('--mode', choices=['train', 'predict'], default='train')
    ap.add_argument('--output_dir', type=str, default='output')

    ap.add_argument('--training_text', type=str, default='Data/x_train.txt')
    ap.add_argument('--training_labels', type=str, default='Data/y_train.txt')
    ap.add_argument('--testing_text', type=str, default='Data/x_test.txt')
    ap.add_argument('--testing_labels', type=str, default='Data/y_test.txt')

    cfg = vars(ap.parse_args())

    print("Parameters:")
    for k, v in cfg.items():
        print("  %12s : %s" % (k, v))

    if cfg['mode'] == 'train':
        train(**cfg)
    elif cfg['mode'] == 'predict':
        raise NotImplementedError()


if __name__ == '__main__':
    main()
