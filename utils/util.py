# Author: Zylo117

import math
import os
import uuid
from glob import glob
from typing import Union

import cv2
import numpy as np
import torch
import webcolors
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
from torchvision.ops.boxes import batched_nms



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



STANDARD_COLORS = [
    'Red', 'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]





def plot_one_box(img, coord, label_index=None, score=None, line_thickness=None):
    '''
    coord = [x1, y1, x2, y2]
    label_index: start from 0
    '''
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    label = class_labels[label_index+1]
    if label_index >= len(color_list):
        label_index = label_index%len(color_list)
    color = color_list[label_index]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    
    if label_index:
        tf = max(tl - 2, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 2, thickness=tf)[0]
        if score:
            s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 2, thickness=tf)[0]
            c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 2, [0, 0, 0],
                        thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)
        else:
            c2 = c1[0] + t_size[0] + 15, c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, '{} GT'.format(label), (c1[0], c1[1] - 2), 0, float(tl) / 2, [0, 0, 0],
                        thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)
    return img



def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard

color_list = standard_to_bgr(STANDARD_COLORS)


