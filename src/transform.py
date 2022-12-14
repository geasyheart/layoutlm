# -*- coding: utf8 -*-
# https://guillaumejaume.github.io/FUNSD/
# form understanding in noisy scanned documents
import os.path
from typing import List

import cv2
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from src.conf import DATA_PATH
from src.g import tokenizer


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, boxes, actual_bboxes, file_name, page_size):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


class InputFeatures(object):
    def __init__(
            self,
            input_ids,
            attention_mask,
            label_ids,
            boxes,
            actual_bboxes,
            file_name,
            page_size,
    ):
        assert (
                0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(boxes)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_ids = label_ids
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size
        self.img_arr = None


def get_labels():
    with open(DATA_PATH.joinpath('labels.txt'), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


class FunsdDataSet(Dataset):
    def __init__(self, mode, device='cpu'):
        examples = read_examples_from_file(DATA_PATH, mode)
        self.features = convert_examples_to_features(
            examples=examples,
            labels=get_labels(),
            pad_token_label_id=-100,
        )
        self.device = device

    def __getitem__(self, index) -> T_co:
        example = self.features[index]
        example.img_arr = torch.from_numpy(load_img(example.file_name))
        return example

    def __len__(self):
        return len(self.features)

    def collate_fn(self, batch_data: List[InputFeatures]):
        input_ids, attention_masks, bbox, labels = [], [], [], []
        img_arrs = []
        for ife in batch_data:
            input_ids.append(torch.tensor(ife.input_ids, dtype=torch.long))
            attention_masks.append(torch.tensor(ife.attention_mask, dtype=torch.bool))
            bbox.append(torch.tensor(ife.boxes, dtype=torch.long))
            labels.append(torch.tensor(ife.label_ids, dtype=torch.long))
            img_arrs.append(ife.img_arr)
        pad_input_ids = pad_sequence(input_ids, padding_value=tokenizer.pad_token_id, batch_first=True)
        pad_token_type_ids = torch.zeros(pad_input_ids.shape, dtype=torch.long)
        pad_attention_mask = pad_sequence(attention_masks, padding_value=tokenizer.pad_token_id, batch_first=True)
        # pad bbox [0, 0, 0, 0]
        pad_bbox = pad_sequence(bbox, padding_value=tokenizer.pad_token_id, batch_first=True)
        pad_labels = pad_sequence(labels, padding_value=-100, batch_first=True)
        return pad_input_ids.to(self.device), pad_token_type_ids.to(self.device), pad_attention_mask.to(
            self.device), pad_bbox.to(self.device), torch.stack(img_arrs).to(self.device), pad_labels.to(self.device)

    def to_dl(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, f'{mode}.txt')
    box_file_path = os.path.join(data_dir, f'{mode}_box.txt')
    image_file_path = os.path.join(data_dir, f'{mode}_image.txt')
    guid_index = 1
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f, open(box_file_path, 'r', encoding='utf-8') as fb, open(
            image_file_path, 'r', encoding='utf-8') as fi:
        words, boxes, actual_bboxes = [], [], []
        file_name, page_size, labels = None, None, []
        for line, bline, iline in zip(f, fb, fi):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                # 表示满足<510一个input了
                if words:
                    examples.append(
                        InputExample(
                            guid=f'{mode}-{guid_index}',
                            words=words,
                            labels=labels,
                            boxes=boxes,
                            actual_bboxes=actual_bboxes,
                            file_name=file_name,
                            page_size=page_size
                        )
                    )
                    guid_index += 1
                    words, boxes, actual_bboxes = [], [], []
                    file_name, page_size, labels = None, None, []
            else:
                splits = line.split('\t')
                bsplits = bline.split('\t')
                isplits = iline.split('\t')
                assert len(splits) == 2
                assert len(bsplits) == 2
                assert len(isplits) == 4
                assert splits[0] == bsplits[0] == isplits[0]

                words.append(splits[0])
                labels.append(splits[-1].replace("\n", ""))
                box = bsplits[-1].replace("\n", "")
                box = [int(b) for b in box.split()]
                boxes.append(box)
                actual_bbox = [int(b) for b in isplits[1].split()]
                actual_bboxes.append(actual_bbox)
                page_size = [int(i) for i in isplits[2].split()]
                file_name = isplits[3].strip()
                file_name = os.path.join(
                    DATA_PATH, 'funsd-dataset', 'dataset',
                    'training_data' if mode == 'train' else 'testing_data',
                    'images',
                    file_name
                )

        if words:
            examples.append(
                InputExample(
                    guid=f'{mode}-{guid_index}',
                    words=words,
                    labels=labels,
                    boxes=boxes,
                    actual_bboxes=actual_bboxes,
                    file_name=file_name,
                    page_size=page_size,
                )
            )
    return examples


def convert_examples_to_features(
        examples: List[InputExample],
        labels: List[str],
        pad_token_label_id=-100,
):
    label_map = {label: i for i, label in enumerate(labels)}
    features = []

    for ex_index, example in enumerate(examples):
        page_size = example.page_size
        width, height = page_size

        tokens, token_boxes, actual_bboxes, label_ids = [], [], [], []
        # https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/layoutlm#transformers.LayoutLMForTokenClassification
        for word, label, box, actual_box in zip(example.words, example.labels, example.boxes, example.actual_bboxes):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            actual_bboxes.extend([actual_box] * len(word_tokens))
            # 第一个token赋予真实标签，剩下的都为交叉熵的ignore_index
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        if len(tokens) > 510:
            tokens = tokens[: 510]
            token_boxes = token_boxes[: 510]
            actual_bboxes = actual_bboxes[: 510]
            label_ids = label_ids[: 510]

        tokens += [tokenizer.sep_token]
        token_boxes += [tokenizer.sep_token_box]
        actual_bboxes += [[0, 0, width, height]]
        label_ids += [pad_token_label_id]

        tokens = [tokenizer.cls_token] + tokens
        token_boxes = [tokenizer.cls_token_box] + token_boxes
        actual_bboxes = [[0, 0, width, height]] + actual_bboxes
        label_ids = [pad_token_label_id] + label_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_ids=label_ids,
                boxes=token_boxes,
                actual_bboxes=actual_bboxes,
                file_name=example.file_name,
                page_size=page_size
            )
        )
    return features


def load_img(img_path):
    # https://huggingface.co/docs/transformers/main/en/model_doc/layoutlmv2#overview
    # tips
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resize_h, resize_w = 224, 224
    im_shape = img.shape[0:2]
    im_scale_y = resize_h / im_shape[0]
    im_scale_x = resize_w / im_shape[1]
    img_new = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=2)
    mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
    std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
    img_new = img_new / 255.0
    img_new -= mean
    img_new /= std
    img = img_new.transpose((2, 0, 1))
    return img.astype('float32')


if __name__ == '__main__':
    for _ in FunsdDataSet(mode='train').to_dl(32, False):
        print()
