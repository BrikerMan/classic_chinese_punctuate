# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: split_data
@time: 2018/11/24

"""
import os
import tqdm
import shutil
import pandas as pd
from utils import helper


def get_files_info(files_path: str):
    file_info = []
    file_list = helper.get_all_files(path=files_path)
    for file in tqdm.tqdm(file_list, desc="generating file info"):
        info = {
            'token_count': 0,
            'chinese_count': 0,
            'mark_count': 0,
            'mark_list': []
        }
        lines = open(file, 'r', encoding='utf-8').read().splitlines()
        for line in lines:
            line = line.strip()
            for char in line:
                if helper.chinese_regex.match(char):
                    info['chinese_count'] += 1
                elif char in helper.TARGET_CHARS:
                    info['mark_count'] += 1
                    info['mark_list'].append(char)

        info['token_count'] = info['chinese_count'] + info['mark_count']
        info['mark_list'] = ' '.join(set(info['mark_list']))
        info['mark_rate'] = info['mark_count'] / info['token_count']
        info['file'] = file.replace(files_path, '')
        file_info.append(info)
    df = pd.DataFrame(file_info)
    df.to_csv('file_info.csv')
    return df


def split_marked_unmarked_files(original_path: str,
                                target_path: str):
    marked_path = os.path.join(target_path, 'marked')
    unmarked_path = os.path.join(target_path, 'unmarked')
    helper.make_dir_if_needs(marked_path)
    helper.make_dir_if_needs(unmarked_path)
    df = get_files_info(original_path)

    # 根据标点符号的比例简单粗暴地分数据，后期要重点优化这个方法
    marked_df = df[df['mark_rate'] >= 0.1]
    unmarked_df = df[df['mark_rate'] < 0.1]
    columns = list(df.columns)

    def copy_files(t_df: pd.DataFrame, copy_to_path: str):
        for file in tqdm.tqdm(t_df.values, desc='copying files to {}'.format(copy_to_path)):
            file_name = file[columns.index('file')]
            target = os.path.join(copy_to_path + file_name)
            helper.make_dir_if_needs(target)
            shutil.copy(os.path.join(original_path + file_name), target)

    copy_files(marked_df, marked_path)
    copy_files(unmarked_df, unmarked_path)


if __name__ == '__main__':
    print("hello, world")