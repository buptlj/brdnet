
import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from libtiff import TIFF


class CTDataset(Dataset):

    def __init__(self, mode='train', data_path='', patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        self.mode = mode
        self.data_path = data_path
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform

        self.inp_paths = []
        self.gt_paths = []

        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.gt_paths.append(line)
                inp_path = line.replace('projections_noisefree', 'projections_gaussian_noise').replace('data_', 'data_noisy_')
                self.inp_paths.append(inp_path)
        print('data num: ', len(self.inp_paths))

    def __len__(self):
        return len(self.inp_paths)

    def __getitem__(self, idx):
        inp_path = self.inp_paths[idx]
        gt_path = self.gt_paths[idx]
        inp_data = get_data(inp_path)
        gt_data = get_data(gt_path)
        if self.patch_size:
            input_patches, target_patches = get_patch(inp_data, gt_data, self.patch_n, self.patch_size)
            #print(input_patches.min(), input_patches.max())
            #print(target_patches.min(), target_patches.max())
            return (input_patches, target_patches)
        else:
            return (inp_data, gt_data)

def get_data(data_path):
    suffix = data_path.split('/')[-1].split('.')[-1]
    if suffix == 'npy':
        data = np.load(data_path)
    elif suffix == 'tif':
        data = TIFF.open(data_path, mode = "r")
        data = list(data.iter_images())[0]
    else:
        data = cv2.imread(data_path)

    min_v = data.min()
    max_v = data.max()
    #input_img = (input_img - min_v) / (max_v - min_v) * 255
    data = (data - min_v) / (max_v - min_v)
    #print('data: ', data.min(), data.max())
    return data

def val_img():
    gt_path = '/data/projects/applect/projections_noisefree/data_31101_520.tif'
    inp_path = gt_path.replace('projections_noisefree', 'projections_gaussian_noise').replace('data_', 'data_noisy_')
    data_name = gt_path.split('/')[-1].split('.')[0]
    input_img = get_data(inp_path)
    target_img = get_data(gt_path)
    return (input_img, target_img, data_name)

def get_patch(full_input_img, full_target_img, patch_n, patch_size, drop_background=0.1):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    n = 0
    while n < patch_n:
        if new_h >= h:
            top = 0
        else:
            top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        patch_input_img = full_input_img[top:top + new_h, left:left + new_w]
        patch_target_img = full_target_img[top:top + new_h, left:left + new_w]

        if (np.mean(patch_input_img) < drop_background) or \
            (np.mean(patch_target_img) < drop_background):
            continue
        else:
            n += 1
            patch_input_imgs.append(patch_input_img)
            patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)

