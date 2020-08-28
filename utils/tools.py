import os
import time
import numpy as np

from libtiff import TIFF
import cv2



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


def get_data(data_path, norm=0):
    suffix = data_path.split('/')[-1].split('.')[-1]
    if suffix == 'npy':
        data = np.load(data_path)
    elif suffix == 'tif':
        data = TIFF.open(data_path, mode = "r")
        data = list(data.iter_images())[0]
    else:
        data = cv2.imread(data_path)

    if norm:
        min_v = data.min()
        max_v = data.max()
        data = (data - min_v) / (max_v - min_v) * norm
    #print('data: ', data.min(), data.max())
    return data.astype(np.float32)


def save_data(save_dir, data_name, data):
    save_path = os.path.join(save_dir, data_name + '.jpg')
    cv2.imwrite(save_path, data)


def norm_data(data):
    min_v = data.min()
    max_v = data.max()
    data = (data - min_v) / (max_v - min_v)
    return data




