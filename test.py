
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import numpy as np
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import BRDNet
from libtiff import TIFF
import cv2
import skimage
import skimage.metrics
from utils.tools import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help="train | test")
    parser.add_argument('--pretrained', type=str, default='./ckpt/b8xn10xs50/model_checkpoint_35.pth', help="pretrained model")

    parser.add_argument('--test_data', type=str, default='./data/')
    parser.add_argument('--save_dir', type=str, default='./ckpt/1')
    parser.add_argument('--result_fig', type=bool, default=True)
    parser.add_argument('--transform', type=bool, default=False)

    # if patch training, batch size is (--patch_n x --batch_size)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--test_interval', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    return args


def denorm(data, data_path):
    org_gt = get_data(data_path)
    max_v = org_gt.max()
    data = data * max_v
    return data


def save_denoise_img(pred, data_path, save_img=True):
    #'/data/projects/applect/projections_gaussian_noise/data_noisy_31116_090.tif'
    save_path = data_path.replace('prj_scattering', 'prj_scattering_denoise').replace('_noisy', '').replace('.tif', '.npy')
    np.save(save_path, pred)
    if save_img:
        min_v = pred.min()
        max_v = pred.max()
        pred = (pred - min_v) / (max_v - min_v) * 255
        cv2.imwrite(save_path.replace('.npy', '.jpg'), pred)


def main():
    args = get_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    device = args.device

    model = BRDNet()
    model.load_state_dict(torch.load(args.pretrained))
    model.to(device)
    model.eval()
    
    data_list = get_file_list(args.test_data, suffix=['npy', 'tif'])
    eval_psnr = 0
    org_avg_psnr = 0
    with torch.no_grad():
        for inp_path in tqdm(data_list):
            gt_path = inp_path.replace('prj_scattering', 'org_prj')
            org_data = get_data(inp_path, norm=1)
            inp_data = torch.from_numpy(org_data).float()
            inp_data = inp_data.unsqueeze(0).unsqueeze(0).to(args.device)
            pred = model(inp_data)
            pred[pred < 0] = 0
            pred[pred > 1] = 1

            pred = pred.squeeze(0).squeeze(0).cpu().numpy()
            save_denoise_img(pred, inp_path, save_img=True)
            gt_data = get_data(gt_path, norm=1)
            org_psnr = skimage.metrics.peak_signal_noise_ratio(org_data, gt_data, data_range=1)
            psnr = skimage.metrics.peak_signal_noise_ratio(pred, gt_data, data_range=1)
            eval_psnr += psnr
            org_avg_psnr += org_psnr
    avg_psnr = eval_psnr / len(data_list)
    org_avg_psnr = org_avg_psnr / len(data_list)
    print('total data: {}, avg psnr: {}'.format(len(data_list), avg_psnr))
    print('total data: {}, org psnr: {}'.format(len(data_list), org_avg_psnr))


if __name__ == '__main__':
    main()


