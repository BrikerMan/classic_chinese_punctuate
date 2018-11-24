# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: helper
@time: 2018/11/24

"""
import os
import re
import keras
import h5py
import pathlib
import numpy as np
from typing import List
from keras import backend as K
from keras.utils import to_categorical

from utils import macros


chinese_regex = re.compile('[\u4e00-\u9fa5]')

TARGET_CHARS = ['，', '。', '】', '【', '、', '：', '“', '”', '；', '》', '《', '○', '）', '（', '？']


def get_all_files(path: str, file_type: str='txt') -> List[str]:
    """
    get all the files from given folder and it's sub folders
    :param path: target folder
    :param file_type: target file type
    :return: file list
    """
    text_files = list()

    for root, dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1].lower() == ".{}".format(file_type):
                text_files.append(os.path.join(root, f))
    return text_files


def make_dir_if_needs(path):
    p = os.path.split(path)
    pathlib.Path(p[0]).mkdir(parents=True, exist_ok=True)


def format_line(text):
    """
    split one line text to x array (chinese word list) and y array (labels for words)
    :param text: target line text
    :return: Tuple[x array, y array]
    """
    text = text
    target_x = []
    target_label = []
    for char in text:
        if chinese_regex.match(char):
            target_x.append(char)
            target_label.append(macros.NO_TAG)
        elif char in TARGET_CHARS and len(target_label) > 0:
            target_label[-1] = char
    return target_x, target_label


def h5f_generator(h5path: str,
                  indices: List[int],
                  label_count: int,
                  batch_size: int=128):
    """
    fit generator for h5 file
    :param h5path: target f5file
    :param indices: target index list
    :param label_count: label counts to covert y label to one hot array
    :param batch_size:
    :return:
    """

    db = h5py.File(h5path, "r")

    while True:
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_indices.sort()
            x = db["x"][batch_indices, :]
            y = to_categorical(db["y"][batch_indices, :],
                               num_classes=label_count,
                               dtype=np.int)
            yield (x, y)


def predict_with_model(tokenizer,
                       text,
                       model: keras.models.Model):
    input_text = [i for i in text if chinese_regex.match(i)]
    input_text = input_text[:tokenizer.max_length-2]
    input_token = tokenizer.tokenize(input_text)
    input_x = keras.preprocessing.sequence.pad_sequences([input_token],
                                                         maxlen=tokenizer.max_length,
                                                         padding='post')
    predict_idx = model.predict(input_x)[0].argmax(1)
    labels = tokenizer.label_de_tokenize(predict_idx, length=len(input_text))
    final = ''
    for i in range(len(input_text)):
        final += input_text[i]
        if labels[i] != 'O':
            final += '{}'.format(labels[i])
    return final


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


if __name__ == '__main__':
    print("hello, world")