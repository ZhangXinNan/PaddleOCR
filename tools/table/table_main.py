
import os
import argparse
import numpy as np
import cv2
import random

from line_seg import line_length
_CHAR_SIZE = 32


def draw_lines(img, lines_hor, lines_ver):
    img_show = img.copy()
    for line in lines_hor:
        line = line[0].astype(np.int)
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        cv2.line(img_show, (x1, y1), (x2, y2), (255, 0, 0), 3)
    for line in lines_ver:
        line = line[0].astype(np.int)
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        cv2.line(img_show, (x1, y1), (x2, y2), (0, 255, 0), 3)
    img_show = cv2.addWeighted(img, 0.5, img_show, 0.5, 0)
    return img_show


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

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gaus = cv2.GaussianBlur(img_gray, (5, 5), 0)

            fld = cv2.ximgproc.createFastLineDetector()
            lines = fld.detect(img_gaus)
            print(lines)
            # lines_valid = []
            lines_valid_hor, lines_valid_ver = [], []
            for i, line in enumerate(lines):
                line = line[0].astype(np.int)
                x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
                # 大于文本高度1.5倍的才算有效的直线
                # if img_mask is not None:
                if line_length(x1, y1, x2, y2) > _CHAR_SIZE * 1.5:
                    # lines_valid.append(lines[i])
                    if abs(x1 - x2) > 10 * abs(y1 - y2):
                        lines_valid_hor.append(lines[i])
                    elif abs(y1 - y2) > 10 * abs(x1 - x2):
                        lines_valid_ver.append(lines[i])
            img_show_lines = draw_lines(img, lines_valid_hor, lines_valid_ver)
            cv2.imwrite(os.path.join(args.out_dir, filename + '.lines.jpg'), img_show_lines)


if __name__ == '__main__':
    main(get_args())

