import glob
import math
import os
import random
import shutil
import subprocess
import time
import logging
from contextlib import contextmanager
from copy import copy
from pathlib import Path
import platform
from statistics import stdev

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from scipy.cluster.vq import kmeans
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from utils.torch_utils import init_seeds, is_parallel
from utils.datasets import letterbox
from torchvision.transforms import transforms
from PIL import Image


trans = transforms.Compose(
    [transforms.Resize((28,28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])])


char_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'K': 18, 'L': 19, 'M': 20, 'N': 21, 'P': 22, 'R': 23, 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'X': 28, 'Y': 29, 'Z': 30}


def img_for_train(img0, img_size=640):
    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img


def crop_plate(x, img):
    # Crop plate by coordinate
    x0, x1, x2, x3 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
    crop_img = img[x1:x3, x0:x2]
    return crop_img


def crop_char(x, img):
    # Crop char and return bbox
    x0, x1, x2, x3 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
    bbox = {0: int(x[0]), 1: int(x[1]),2: int(x[2]),3: int(x[3])}
    crop_img = img[x1:x3, x0:x2]
    return {'img':crop_img, 'bbox': bbox}


def get_line(blocks):
    list_y0 = []
    new_blocks = []
    for block in blocks:
        list_y0.append(block['bbox'][1])
    for block in blocks:
        if block['bbox'][1] < (min(list_y0) + 10):
            block['line'] = min(list_y0)
        elif block['bbox'][1] > (max(list_y0) - 10):
            block['line'] = max(list_y0)
        else:
            block['line'] = block['bbox'][1]
        new_blocks.append(block)
    return new_blocks


def sort_blocks(blocks):
    sblocks = []
    blocks = get_line(blocks)
    for block in blocks:
        x0 = str(int(block['bbox'][0]+0.99999)).rjust(4,"0")
        y0 = str(int(block['line']+0.99999)).rjust(4,"0")
        sortkey = y0 + x0
        sblocks.append([sortkey, block])
    sblocks.sort(key=lambda x: x[0], reverse=False)
    return [block[1] for block in sblocks]


def transform_char(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = trans(image)
    return image.view(1, 1, 28, 28)


def get_key(val): 
    for key, value in char_dict.items(): 
         if val == value: 
             return key
    return "key doesn't exist"
