# -*- coding: utf8 -*-
#
import json
import os
from typing import List, Dict

from PIL import Image
from src.g import tokenizer
from src.conf import DATA_PATH


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


def bbox_string(bbox, width, height):
    return " ".join([str(i) for i in normalize_bbox(bbox=bbox, width=width, height=height)])


def actual_bbox_string(box, width, length):
    return str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + "\t" + str(width) + " " + str(
        length)


def convert(mode='train'):
    with open(DATA_PATH.joinpath(f'{mode}.txt.tmp'), 'w', encoding='utf-8') as fw, \
            open(DATA_PATH.joinpath(f'{mode}_box.txt.tmp'), 'w', encoding='utf-8') as fbw, \
            open(DATA_PATH.joinpath(f'{mode}_image.txt.tmp'), 'w', encoding='utf-8') as fiw:

        path = DATA_PATH.joinpath('funsd-dataset').joinpath('dataset')
        if mode == 'train':
            annotations_path = path.joinpath('training_data').joinpath('annotations')
        else:
            annotations_path = path.joinpath('testing_data').joinpath('annotations')
        for file in os.listdir(annotations_path):
            file_path = str(annotations_path.joinpath(file))
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            image_path = file_path.replace("annotations", "images")
            image_path = image_path.replace('json', 'png')
            file_name = os.path.basename(image_path)
            image = Image.open(image_path)
            width, height = image.size
            for item in data['form']:
                # 看他child
                words: List[Dict] = item['words']
                label: str = item['label']

                words = [w for w in words if w["text"].strip() != ""]

                if len(words) == 0: continue
                if label == 'other':
                    for w in words:
                        fw.write(f'{w["text"]}\tO\n')
                        fbw.write(f'{w["text"]}\t{bbox_string(w["box"], width, height)}\n')
                        fiw.write(f'{w["text"]}\t{actual_bbox_string(w["box"], width, height)}\t{file_name}\n')
                else:
                    if len(words) == 1:
                        fw.write(f"{words[0]['text']}\tS-{label.upper()}\n")
                        fbw.write(f'{words[0]["text"]}\t{bbox_string(words[0]["box"], width, height)}\n')
                        fiw.write(
                            f'{words[0]["text"]}\t{actual_bbox_string(words[0]["box"], width, height)}\t{file_name}\n')
                    else:
                        fw.write(f"{words[0]['text']}\tB-{label.upper()}\n")
                        fbw.write(f'{words[0]["text"]}\t{bbox_string(words[0]["box"], width, height)}\n')
                        fiw.write(
                            f'{words[0]["text"]}\t{actual_bbox_string(words[0]["box"], width, height)}\t{file_name}\n')
                        for w in words[1:-1]:
                            fw.write(f'{w["text"]}\tI-{label.upper()}\n')
                            fbw.write(f'{w["text"]}\t{bbox_string(w["box"], width, height)}\n')
                            fiw.write(f'{w["text"]}\t{actual_bbox_string(w["box"], width, height)}\t{file_name}\n')
                        fw.write(f'{words[-1]["text"]}\tE-{label.upper()}\n')
                        fbw.write(f'{words[-1]["text"]}\t{bbox_string(words[-1]["box"], width, height)}\n')
                        fiw.write(
                            f'{words[-1]["text"]}\t{actual_bbox_string(words[-1]["box"], width, height)}\t{file_name}\n')
            fw.write('\n')
            fbw.write('\n')
            fiw.write('\n')


def seg_file(file_path):
    subword_len_counter = 0
    output_path = str(file_path)[:-4]
    with open(file_path, 'r', encoding='utf-8') as f_p, open(output_path, 'w', encoding='utf-8') as fw_p:
        for line in f_p:
            line = line.rstrip()
            if not line:
                # 表示这个文件已经读完
                fw_p.write(f'{line}\n')
                subword_len_counter = 0
                continue
            token = line.split('\t')[0]
            current_subwords_len = len(tokenizer.tokenize(token))
            if subword_len_counter + current_subwords_len > 510:  # cls sep
                # 就将这个文件分成多份
                fw_p.write("\n")
                fw_p.write(f'{line}\n')
                subword_len_counter = current_subwords_len
                continue
            subword_len_counter += current_subwords_len
            fw_p.write(line + "\n")


def gen_label():
    labels = set()
    with open(DATA_PATH.joinpath('train.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:continue
            word, label = line.split('\t')
            labels.add(label)
    with open(DATA_PATH.joinpath('labels.txt'), 'w', encoding='utf-8') as f:
        f.writelines([f'{l}\n' for l in labels])


if __name__ == '__main__':
    convert(mode='train')
    seg_file(DATA_PATH.joinpath("train.txt.tmp"))
    seg_file(DATA_PATH.joinpath('train_box.txt.tmp'))
    seg_file(DATA_PATH.joinpath("train_image.txt.tmp"))

    convert(mode='test')
    seg_file(DATA_PATH.joinpath("test.txt.tmp"))
    seg_file(DATA_PATH.joinpath('test_box.txt.tmp'))
    seg_file(DATA_PATH.joinpath("test_image.txt.tmp"))

    gen_label()
