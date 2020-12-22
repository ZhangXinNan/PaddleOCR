
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='')
    parser.add_argument('--outdir', default='')
    return parser.parse_args()


def main(args):
    pass


if __name__ == '__main__':
    main(get_args())

