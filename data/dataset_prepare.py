import os, sys
import shutil

import os
import random
import argparse
import time

def dataset_arrange():
    dataset_dir = '/home/hongrui/project/dataset/stanfordDogsDataset'
    ori_img_dataset = os.path.join(dataset_dir, 'Images')
    ori_anno_dataset = os.path.join(dataset_dir, 'Annotation')

    splits = ['train', 'val', 'test']

    for split in splits:
        first_out_dir = os.path.join(dataset_dir, split)
        os.makedirs(first_out_dir, exist_ok=True)
    sub_dirs = os.listdir(ori_img_dataset)
    # print('sub_dirs', sub_dirs)


    for sub_dir in sub_dirs:
        real_img_dir = os.path.join(ori_img_dataset, sub_dir)
        img_names = os.listdir(real_img_dir)
        # print('img_names', len(img_names))
        random.shuffle(img_names)
        total_num = len(img_names)
        for i, img_name in enumerate(img_names):
            ori_img_filepath = os.path.join(real_img_dir, img_name)
            ori_anno_filepath = ori_img_filepath.replace('Images', 'Annotation').split('.')[0]
            # print(ori_img_filepath)
            # print(ori_anno_filepath)
            # return
            if i < 0.7*total_num:
                split = 'train'
            elif i < 0.85*total_num:
                split = 'val'
            else:
                split = 'test'
            first_out_dir = os.path.join(dataset_dir, split)
            output_img_dir = os.path.join(first_out_dir, 'Images', sub_dir)
            os.makedirs(output_img_dir, exist_ok=True)
            output_anno_dir = os.path.join(first_out_dir, 'Annotation', sub_dir)
            os.makedirs(output_anno_dir, exist_ok=True)

            out_img_filepath = os.path.join(output_img_dir, img_name)
            out_anno_filepath = out_img_filepath.replace('Images', 'Annotation').split('.')[0]
            shutil.move(ori_img_filepath, out_img_filepath)
            shutil.move(ori_anno_filepath, out_anno_filepath)
            # return



if __name__ == '__main__':
    dataset_arrange()


