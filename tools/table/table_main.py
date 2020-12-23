
import os
import argparse
import numpy
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./imgs')
    parser.add_argument('--out_dir', default='./imgs_result')
    return parser.parse_args()


def main(args):
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if os.path.isfile(args.input):
        img = cv2.imread(args.input)
        print(img.shape)
    elif os.path.isdir(args.input):
        for filename in os.listdir(args.input):
            name, suffix = os.path.splitext(filename)
            if suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            img_path = os.path.join(args.input, filename)
            img = cv2.imread(img_path)
            print(img_path, img.shape)


if __name__ == '__main__':
    main(get_args())

