import numpy as np 
import matplotlib.pyplot as plt
import cv2
import math
from scipy.fftpack import fft, fftshift, ifft
import random
from tqdm import tqdm
import os
from PIL import Image
from libtiff import TIFF


def get_file_list(file_dir, all_data=False, suffix=['jpg', 'jpeg', 'JPG', 'JPEG', 'png']):
    if not os.path.exists(file_dir):
        print('path {} is not exist'.format(file_dir))
        return []
    img_list = []

    for root, sdirs, files in os.walk(file_dir):
        if not files:
            continue
        for filename in files:
            filepath = os.path.join(root, filename)
            if all_data or filename.split('.')[-1] in suffix:
                img_list.append(filepath)
    return img_list


if __name__ == "__main__":
    gauss_noise_prj = '/data/projects/applect/projections_gaussian_noise/'
    gt_prj = '/data/projects/applect/projections_noisefree/'

    img_list = get_file_list(gt_tif_dir, suffix=['tif'])
    items = []
    for tif_path in tqdm(img_list):
        #img_path = 'data_31101_520_0.jpg'
        img_name = tif_path.split('/')[-1].split('.')[0]
        rec_prj_path = os.path.join(rec_prj_dir, 'data_'+img_name+'.npy')
        if not os.path.exists(rec_prj_path):
            print('not exist: ', rec_prj_path)
        item = '{} {}\n'.format(rec_prj_path, tif_path)
        items.append(item)

    with open('img_list.txt', 'w') as f:
        f.write(''.join(items))
    print('done')


