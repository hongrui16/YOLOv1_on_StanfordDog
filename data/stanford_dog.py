import os
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO




class_labels = {1: 'Chihuahua', 2: 'Japanese_spaniel', 3: 'Maltese_dog', 4: 'Pekinese', 5: 'Shih-Tzu', 
    6: 'Blenheim_spaniel', 7: 'papillon', 8: 'toy_terrier', 9: 'Rhodesian_ridgeback', 10: 'Afghan_hound', 
    11: 'basset', 12: 'beagle', 13: 'bloodhound', 14: 'bluetick', 15: 'black-and-tan_coonhound', 
    16: 'Walker_hound', 17: 'English_foxhound', 18: 'redbone', 19: 'borzoi', 20: 'Irish_wolfhound', 
    21: 'Italian_greyhound', 22: 'whippet', 23: 'Ibizan_hound', 24: 'Norwegian_elkhound', 25: 'otterhound', 
    26: 'Saluki', 27: 'Scottish_deerhound', 28: 'Weimaraner', 29: 'Staffordshire_bullterrier', 30: 'American_Staffordshire_terrier', 
    31: 'Bedlington_terrier', 32: 'Border_terrier', 33: 'Kerry_blue_terrier', 34: 'Irish_terrier', 35: 'Norfolk_terrier', 
    36: 'Norwich_terrier', 37: 'Yorkshire_terrier', 38: 'wire-haired_fox_terrier', 39: 'Lakeland_terrier', 40: 'Sealyham_terrier', 
    41: 'Airedale', 42: 'cairn', 43: 'Australian_terrier', 44: 'Dandie_Dinmont', 45: 'Boston_bull', 
    46: 'miniature_schnauzer', 47: 'giant_schnauzer', 48: 'standard_schnauzer', 49: 'Scotch_terrier', 50: 'Tibetan_terrier', 
    51: 'silky_terrier', 52: 'soft-coated_wheaten_terrier', 53: 'West_Highland_white_terrier', 54: 'Lhasa', 55: 'flat-coated_retriever', 
    56: 'curly-coated_retriever', 57: 'golden_retriever', 58: 'Labrador_retriever', 59: 'Chesapeake_Bay_retriever', 60: 'German_short-haired_pointer', 
    61: 'vizsla', 62: 'English_setter', 63: 'Irish_setter', 64: 'Gordon_setter', 65: 'Brittany_spaniel', 
    66: 'clumber', 67: 'English_springer', 68: 'Welsh_springer_spaniel', 69: 'cocker_spaniel', 70: 'Sussex_spaniel', 
    71: 'Irish_water_spaniel', 72: 'kuvasz', 73: 'schipperke', 74: 'groenendael', 75: 'malinois', 
    76: 'briard', 77: 'kelpie', 78: 'komondor', 79: 'Old_English_sheepdog', 80: 'Shetland_sheepdog', 
    81: 'collie', 82: 'Border_collie', 83: 'Bouvier_des_Flandres', 84: 'Rottweiler', 85: 'German_shepherd', 
    86: 'Doberman', 87: 'miniature_pinscher', 88: 'Greater_Swiss_Mountain_dog', 89: 'Bernese_mountain_dog', 90: 'Appenzeller', 
    91: 'EntleBucher', 92: 'boxer', 93: 'bull_mastiff', 94: 'Tibetan_mastiff', 95: 'French_bulldog', 
    96: 'Great_Dane', 97: 'Saint_Bernard', 98: 'Eskimo_dog', 99: 'malamute', 100: 'Siberian_husky', 
    101: 'affenpinscher', 102: 'basenji', 103: 'pug', 104: 'Leonberg', 105: 'Newfoundland', 
    106: 'Great_Pyrenees', 107: 'Samoyed', 108: 'Pomeranian', 109: 'chow', 110: 'keeshond', 
    111: 'Brabancon_griffon', 112: 'Pembroke', 113: 'Cardigan', 114: 'toy_poodle', 115: 'miniature_poodle', 
    116: 'standard_poodle', 117: 'Mexican_hairless', 118: 'dingo', 119: 'dhole', 120: 'African_hunting_dog'}

stanford_dog_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 
96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]

class_names = []
ids = class_labels.keys()
for i in ids:
    class_names.append(class_labels[i])
    
class StanfordDogDataset(Dataset):
    """
    StanfordDog dataset class.
    """
    def __init__(self, data_dir='COCO', 
                 image_set='train', 
                 img_size=None,
                 transform=None, 
                 ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file image_set
            image_set (str): COCO data image_set (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.data_dir = data_dir
        # self.json_file = json_file
        self.coco = COCO(os.path.join(self.data_dir, image_set, 'anno_json', f'{image_set}.json'))
        self.ids = self.coco.getImgIds()
        # self.class_ids = sorted(self.coco.getCatIds())
        self.class_ids = class_names
        self.image_set = image_set
        self.max_labels = 120
        self.img_size = img_size
        self.transform = transform


    def __len__(self):
        return len(self.ids)


    def pull_image(self, index):
        id_ = self.ids[index]
        image_info = self.coco.loadImgs(id_)[0]
        path = os.path.join(self.data_dir, self.image_set, 'Images', image_info['file_name'])
        img = cv2.imread(path)

        return img, id_


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt


    def pull_item(self, index):
        id_ = self.ids[index]

        # anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        anno_ids = self.coco.getAnnIds(imgIds=id_, iscrowd=False)

        annotations = self.coco.loadAnns(anno_ids)

        # load image and preprocess
        image_info = self.coco.loadImgs(id_)[0]
        path = os.path.join(self.data_dir, self.image_set, 'Images', image_info['file_name'])
        img = cv2.imread(path)
        

        assert img is not None

        height, width, channels = img.shape
        
        # COCOAnnotation Transform
        # start here :
        target = []
        for anno in annotations:
            x1 = np.max((0, anno['bbox'][0]))
            y1 = np.max((0, anno['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, anno['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, anno['bbox'][3] - 1))))
            if anno['area'] > 0 and x2 >= x1 and y2 >= y1:
                label_ind = anno['category_id'] - 1
                cls_id = int(label_ind)
                x1 /= width
                y1 /= height
                x2 /= width
                y2 /= height

                target.append([x1, y1, x2, y2, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
        # end here .

        # data augmentation
        if self.transform is not None:
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)

            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


if __name__ == "__main__":
    from transform import Augmentation, BaseTransform

    img_size = 640
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    data_root = '/mnt/share/ssd2/dataset/COCO'
    transform = Augmentation(img_size, pixel_mean, pixel_std)
    transform = BaseTransform(img_size, pixel_mean, pixel_std)

    # img_size = 640
    # dataset = COCODataset(
    #             data_dir=data_root,
    #             img_size=img_size,
    #             transform=transform
    #             )
    
    # for i in range(1000):
    #     im, gt, h, w = dataset.pull_item(i)

    #     # to numpy
    #     image = im.permute(1, 2, 0).numpy()
    #     # to BGR
    #     image = image[..., (2, 1, 0)]
    #     # denormalize
    #     image = (image * pixel_std + pixel_mean) * 255
    #     # to 
    #     image = image.astype(np.uint8).copy()

    #     # draw bbox
    #     for box in gt:
    #         xmin, ymin, xmax, ymax, _ = box
    #         xmin *= img_size
    #         ymin *= img_size
    #         xmax *= img_size
    #         ymax *= img_size
    #         image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
    #     cv2.imshow('gt', image)
    #     cv2.waitKey(0)
