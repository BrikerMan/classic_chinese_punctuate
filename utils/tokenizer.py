# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: tokenizer
@time: 2018/11/24

"""
import os
import random
import json
import h5py
import numpy as np
import tqdm
from typing import List, Dict
from keras.preprocessing import sequence
from utils.embedding import Word2Vec
from utils.macros import PAD, BOS, EOS, UNK, NO_TAG
from utils import helper


class Tokenizer(object):

    PAD = PAD
    BOS = BOS
    EOS = EOS
    UNK = UNK
    NO_TAG = NO_TAG

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    NO_TAG_ID = 1

    def __init__(self):
        self.url = ''

        self.word2idx = {}
        self.idx2word = {}

        self.labels2idx = {}
        self.idx2labels = {}

        self.max_length = 100

        self.w2v = None

    def class_weights(self):
        base_weight = {
            helper.macros.PAD: 0.01,
            helper.macros.NO_TAG: 0.7
        }
        weights = [base_weight.get(i, 1) for i in self.labels2idx]
        return np.asarray(weights)

    def build(self,
              corpus_path: str,
              tokenizer_path: str,
              label_only=False,
              min_accor=3):

        if not label_only:
            file_list = helper.get_all_files(corpus_path)
            word2count = {}
            for file in tqdm.tqdm(file_list, 'building tokens'):
                lines = open(file, 'r', encoding='utf-8').read().splitlines()
                for line in lines:
                    x, _ = helper.format_line(line)
                    for word in line:
                        word2count[word] = word2count.get(word, 0) + 1

            self.word2idx = {
                Tokenizer.PAD: Tokenizer.PAD_ID,
                Tokenizer.BOS: Tokenizer.BOS_ID,
                Tokenizer.EOS: Tokenizer.EOS_ID,
                Tokenizer.UNK: Tokenizer.UNK_ID,
            }
            sorted_word2count = [(k, word2count[k]) for k in sorted(word2count, key=word2count.get, reverse=True)]
            for word, count in sorted_word2count:
                if count >= min_accor:
                    self.word2idx[word] = len(self.word2idx)

        label2count = {
            helper.macros.PAD: 0,
            helper.macros.NO_TAG: 1
        }
        for mark in helper.TARGET_CHARS:
            label2count[mark] = len(label2count)

        self.labels2idx = {
            Tokenizer.PAD: Tokenizer.PAD_ID,
            Tokenizer.NO_TAG: Tokenizer.NO_TAG_ID
        }
        for k, v in label2count.items():
            if k not in self.labels2idx:
                self.labels2idx[k] = len(self.labels2idx)
        helper.make_dir_if_needs(os.path.join(tokenizer_path, 'word2idx.json'))
        if not label_only:
            with open(os.path.join(tokenizer_path, 'word2idx.json'), 'w', encoding='utf-8') as w2idx:
                w2idx.write(json.dumps(self.word2idx, indent=2, ensure_ascii=False))

        with open(os.path.join(tokenizer_path, 'labels2idx.json'), 'w', encoding='utf-8') as l2idx:
            l2idx.write(json.dumps(self.labels2idx, indent=2, ensure_ascii=False))

        print('-------- tokenize finished ----------')
        print('word count : {}'.format(len(self.word2idx)))
        print('label count: {}'.format(len(self.labels2idx)))
        print('use tokenizer by `tokenizer.load(\'{}\')`'.format(tokenizer_path))
        print('-------- tokenize finished ----------')

    def load(self, tokenizer_path):
        self.word2idx = json.load(open(os.path.join(tokenizer_path, 'word2idx.json'), 'r', encoding='utf-8'))
        self.labels2idx = json.load(open(os.path.join(tokenizer_path, 'labels2idx.json'), 'r', encoding='utf-8'))

        self.idx2word = dict([(v, k) for (k, v) in self.word2idx.items()])
        self.idx2labels = dict([(v, k) for (k, v) in self.labels2idx.items()])

    def load_gensim(self, w2v_path):
        self.w2v = Word2Vec()
        self.w2v.load_gensim(w2v_path)
        self.word2idx = self.w2v.word2idx
        self.idx2word = self.w2v.idx2word

        self.labels2idx = json.load(open(os.path.join(w2v_path, 'labels2idx.json'), 'r', encoding='utf-8'))
        self.idx2labels = dict([(v, k) for (k, v) in self.labels2idx.items()])

    def tokenize(self, text, padding=True) -> List[int]:
        tokens = []
        for char in text:
            tokens.append(self.word2idx.get(char, Tokenizer.UNK_ID))
        if padding:
            tokens = [Tokenizer.BOS_ID] + tokens + [Tokenizer.EOS_ID]
        return tokens

    def de_tokenize(self, tokens: List[int], remove_padding=True) -> List[str]:
        text = []
        for token in tokens:
            text.append(self.idx2word[token])
        if remove_padding:
            if text[-1] == Tokenizer.EOS:
                text = text[:-1]
            if text[0] == Tokenizer.BOS:
                text = text[1:]
        return text

    def label_tokenize(self, labels, padding=True) -> List[int]:
        tokens = []
        for char in labels:
            tokens.append(self.labels2idx[char])
        if padding:
            tokens = [Tokenizer.NO_TAG_ID] + tokens + [Tokenizer.NO_TAG_ID]
        return tokens

    def label_de_tokenize(self,
                          tokens: List[int],
                          remove_padding: bool=True,
                          length: int=None) -> List[str]:
        text = []
        if length:
            tokens = tokens[:length+2]
        for token in tokens:
            text.append(self.idx2labels[token])
        if remove_padding:
            text = text[1:-1]
        return text

    def tokenize_files(self, files_path, data_path) -> Dict:
        h5_path = os.path.join(data_path, 'dataset.h5')
        h5 = h5py.File(h5_path, 'a')
        data_info = {
            'length': []
        }
        try:
            h5.create_dataset('x',
                              shape=(500, self.max_length),
                              maxshape=(None, self.max_length),
                              dtype=np.int32,
                              chunks=True)
            h5.create_dataset('y',
                              shape=(500, self.max_length),
                              maxshape=(None, self.max_length),
                              dtype=np.int32,
                              chunks=True)
        except:
            pass

        current_index = 0
        for file in tqdm.tqdm(helper.get_all_files(files_path),
                              desc='processing files'):
            x_padded, y_padded, x_list, y_list = self.process_by_file(file)
            for item in x_list:
                data_info['length'].append(len(item))
            new_index = current_index + len(x_padded)
            if new_index > 500:
                h5['x'].resize((new_index, self.max_length))
                h5['y'].resize((new_index, self.max_length))
            h5['x'][current_index:new_index] = x_padded
            h5['y'][current_index:new_index] = y_padded
            current_index = new_index

        sample_index = random.randint(0, len(h5['x']))
        print('-------- tokenize data finished --------')
        print('dataset path : {}'.format(os.path.abspath(h5_path)))
        print('sample x     : {}'.format(h5['x'][sample_index]))
        print('sample y     : {}'.format(h5['y'][sample_index]))
        print('----------------------------------------')
        h5.close()
        return data_info

    def process_by_file(self, file_path, min_lengh=8):
        lines = open(file_path, 'r', encoding='utf-8').read().splitlines()
        x_list = []
        y_list = []
        for line in lines:
            line = line.strip()
            if line:
                x, y = format_line(line)
                if len(x) == len(y) and len(x) > 8:
                    x_list.append(self.tokenize(x))
                    y_list.append(self.label_tokenize(y))
        x_padded = sequence.pad_sequences(x_list, maxlen=self.max_length, padding='post')
        y_padded = sequence.pad_sequences(y_list, maxlen=self.max_length, padding='post')
        return x_padded, y_padded, x_list, y_list


def format_line(text):
    """
    格式化一行数据
    :param text:
    :return:
    """
    text = text
    target_x = []
    target_label = []
    for char in text:
        if helper.chinese_regex.match(char):
            target_x.append(char)
            target_label.append('O')
        elif char in helper.TARGET_CHARS and len(target_label) > 0:
            target_label[-1] = char
    return target_x, target_label


if __name__ == '__main__':
    print("hello, world")