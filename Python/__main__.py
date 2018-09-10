#!/usr/bin/env python3

import argparse
from Python.train import train


def main():

    ap = argparse.ArgumentParser(description="a Language Identification model")
    ap.add_argument('--mode', choices=['train', 'predict'], default='train')
    ap.add_argument('--output_dir', type=str, default='output')

    cfg = vars(ap.parse_args())

    print("Config:")
    for k, v in cfg.items():
        print("  %12s : %s" % (k, v))

    if cfg['mode'] == 'train':
        train(**cfg)
    elif cfg['mode'] == 'predict':
        raise NotImplementedError()


if __name__ == '__main__':
    main()