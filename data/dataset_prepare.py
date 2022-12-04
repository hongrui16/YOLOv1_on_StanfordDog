import os, sys
import shutil

import os
import random
import argparse
import time
import re

import json
import datetime
import cv2
import numpy as np
from scipy import ndimage as nd
import imageio as io
import xml.etree.ElementTree as ET

import pycocotools
from pycocotools.mask import encode
import pycocotools.coco as coco
from pycocotools.coco import COCO

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



def dump_json():
    dataset_dir = '/home/hongrui/project/dataset/stanfordDogsDataset'
    

    splits = ['train', 'val', 'test']

    
    
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year='2022',
            contributor=None,
            date_created='2022',
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        # images=images,
        # annotations=annotations,
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )  
    img_dir = os.path.join(dataset_dir, 'train', 'Images')
    # ori_anno_dataset = os.path.join(dataset_dir, 'Annotation')
    img_dir_names = os.listdir(img_dir)
    labels = dict()
    for i, cat_name in enumerate(img_dir_names):
        cat_name = cat_name[10:]
        labels[i+1] = cat_name
        
    ids = labels.keys()
    cat_names = []
    for i in ids:
        data["categories"].append(
                dict(supercategory=None, id=i, name=labels[i],)
            )
        cat_names.append(labels[i])
    # print('labels', labels)
    # return

    img_idx = 1
    annot_id = 1
    for split in splits:
        out_anno_dir = os.path.join(dataset_dir, split, 'anno_json')
        os.makedirs(out_anno_dir, exist_ok=True)
        images = []
        annotations = []
        first_dir = os.path.join(dataset_dir, split)
        second_img_dir = os.path.join(first_dir, 'Images')
        # ori_anno_dataset = os.path.join(dataset_dir, 'Annotation')
        cat_dirs = os.listdir(second_img_dir)
        for cat_name in cat_dirs:
            real_img_dir = os.path.join(second_img_dir, cat_name)
            img_names = os.listdir(real_img_dir)
            # print('img_names', len(img_names))
            for i, img_name in enumerate(img_names):
                img_filepath = os.path.join(real_img_dir, img_name)
                anno_filepath = img_filepath.replace('Images', 'Annotation').split('.')[0]
                target = ET.parse(anno_filepath)
                size = target.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                for obj in target.iter('object'):
                    difficult = int(obj.find('difficult').text) == 1
                    if difficult:
                        continue
                    cls_name = obj.find('name').text.strip()
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    h = ymax - ymin
                    w = xmax - xmin
                    bbox = [xmin, ymin, w, h]
                    annotations.append({"segmentation" : [],
                        "area" : np.float(h*w),
                        "iscrowd" : 0,
                        "image_id" : img_idx,
                        "bbox" : bbox,
                        "category_id" : cat_names.index(cls_name) + 1,
                        "id": annot_id})
                    annot_id += 1
                pre_img_name = os.path.join(cat_name, img_name)
                images.append({"date_captured" : "unknown",
                        "file_name" : pre_img_name, 
                        "id" : img_idx,
                        "license" : 1,
                        "url" : "",
                        "height" : height,
                        "width" : width})
                img_idx += 1
                
        data['annotations'] = annotations
        data['images'] = images
         
        out_json_file = os.path.join(out_anno_dir, f'{split}.json')
        if os.path.exists(out_json_file):
            os.remove(out_json_file)
        with open(out_json_file, "w") as f:
            json.dump(data, f)

            
                
if __name__ == '__main__':
    # dataset_arrange()
    dump_json()


