# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import ast

import cv2
from PIL import Image, ImageDraw
import numpy as np
from tools.infer.utility import draw_ocr_box_txt, str2bool, init_args as infer_args


def init_args():
    parser = infer_args()

    # params for output
    parser.add_argument("--output", type=str, default='./output')
    # params for table structure
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_algorithm", type=str, default='TableAttn')
    parser.add_argument("--table_model_dir", type=str)
    parser.add_argument(
        "--merge_no_span_structure", type=str2bool, default=True)
    parser.add_argument(
        "--table_char_dict_path",
        type=str,
        default="../ppocr/utils/dict/table_structure_dict.txt")
    # params for layout
    parser.add_argument("--layout_model_dir", type=str)
    parser.add_argument(
        "--layout_dict_path",
        type=str,
        default="../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt")
    parser.add_argument(
        "--layout_score_threshold",
        type=float,
        default=0.5,
        help="Threshold of score.")
    parser.add_argument(
        "--layout_nms_threshold",
        type=float,
        default=0.5,
        help="Threshold of nms.")
    # params for kie
    parser.add_argument("--kie_algorithm", type=str, default='LayoutXLM')
    parser.add_argument("--ser_model_dir", type=str)
    parser.add_argument(
        "--ser_dict_path",
        type=str,
        default="../train_data/XFUND/class_list_xfun.txt")
    # need to be None or tb-yx
    parser.add_argument("--ocr_order_method", type=str, default=None)
    # params for inference
    parser.add_argument(
        "--mode",
        type=str,
        default='structure',
        help='structure and kie is supported')
    parser.add_argument(
        "--image_orientation",
        type=bool,
        default=False,
        help='Whether to enable image orientation recognition')
    parser.add_argument(
        "--layout",
        type=str2bool,
        default=True,
        help='Whether to enable layout analysis')
    parser.add_argument(
        "--table",
        type=str2bool,
        default=True,
        help='In the forward, whether the table area uses table recognition')
    parser.add_argument(
        "--ocr",
        type=str2bool,
        default=True,
        help='In the forward, whether the non-table area is recognition by ocr')
    # param for recovery
    parser.add_argument(
        "--recovery",
        type=str2bool,
        default=False,
        help='Whether to enable layout of recovery')
    parser.add_argument(
        "--save_pdf",
        type=str2bool,
        default=False,
        help='Whether to save pdf file')

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def draw_structure_result(image, result, font_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    boxes, txts, scores = [], [], []
    for region in result:
        x0, y0 = 0, 0
        if 'bbox' in region:
            x0, y0 = region['bbox'][:2]
        if 'table' == region['type']:
            if 'res' in region and 'boxes' in region['res'] and 'rec_res' in region['res']:
                for box, (txt, score) in zip(region['res']['boxes'], region['res']['rec_res']):
                    x1, y1 = x0 + min(box[0], box[2]), y0 + min(box[1], box[3])
                    x2, y2 = x0 + max(box[0], box[2]), y0 + max(box[1], box[3])
                    box_new = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    boxes.append(box_new)
                    txts.append(txt)
                    scores.append(score)
        else:
            for text_result in region['res']:
                boxes.append(np.array(text_result['text_region']))
                txts.append(text_result['text'])
                scores.append(text_result['confidence'])
    im_show = draw_ocr_box_txt(
        image, boxes, txts, scores, font_path=font_path, drop_score=0)
    # draw table
    h, w = image.height, image.width
    img_right = im_show[:, w:, :].copy()
    for region in result:
        x0, y0 = 0, 0
        if 'bbox' in region:
            x0, y0 = region['bbox'][:2]
        if 'table' == region['type'] and 'cell_bbox' in region['res']:
            for x1, y1, x2, y2, x3, y3, x4, y4 in region['res']['cell_bbox']:
                x1, x2, x3, x4 = x1 + x0, x2 + x0, x3 + x0, x4 + x0
                y1, y2, y3, y4 = y1 + y0, y2 + y0, y3 + y0, y4 + y0
                box = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.fillPoly(img_right, pts=[box], color=color)
    img_right = cv2.addWeighted(im_show[:, w:, :].copy(), 0.5, img_right, 0.5, 0)
    im_show[:, w:, :] = img_right
    return im_show
