# -*- coding: utf8 -*-
#
import os.path


def check(f1, f2):
    with open(f1, 'rb') as f1_obj, open(f2, 'rb') as f2_obj:
        assert f1_obj.read() == f2_obj.read()


if __name__ == '__main__':
    filenames = [
        'train.txt.tmp', 'train_box.txt.tmp', 'train_image.txt.tmp',
        'test.txt.tmp', 'test_box.txt.tmp', 'test_image.txt.tmp',
    ]
    filenames.extend([fn[:-4] for fn in filenames])

    for fn in filenames:
        print(fn)
        p1 = os.path.join('/home/yuzhang/PycharmProjects/layoutlm/data', fn)
        p2 = os.path.join('/home/yuzhang/PycharmProjects/PaddleNLP/examples/multimodal/layoutlm/data', fn)
        check(p1, p2)
