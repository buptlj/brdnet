
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import numpy as np
from torch.backends import cudnn
from torch.utils.data import DataLoader

from dataset import CTDataset
from models import BRDNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.logger import setup_logger
import logging


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help="train | test")
    parser.add_argument('--pretrained', type=str, default='./ckpt/b8xn10xs50/model_checkpoint_35.pth', help="pretrained model")

    parser.add_argument('--train_path', type=str, default='../data/prj_denoise/train.txt')
    parser.add_argument('--val_path', type=str, default='../data/prj_denoise/val.31101.txt')
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

def save_denoise_img(pred, data_path):
    #'/data/projects/applect/projections_gaussian_noise/data_noisy_31116_090.tif'
    pred = pred.squeeze(0).squeeze(0)
    pred = pred.cpu().numpy()
    save_path = data_path.replace('projections_gaussian_noise', 'apple_31101/gauss_prj_denoise').replace('_noisy', '').replace('.tif', '.npy')

    np.save(save_path, pred)


def val(dataloader, model, loss_func, epoch, args, logger):
    total_loss = 0.0
    total_psnr = 0.0
    total_org_psnr = 0.0
    model.eval()
    with torch.no_grad():
        for batch_i, (inp_data, gt_data, inp_path) in enumerate(dataloader):
            inp_data = inp_data.unsqueeze(0).to(args.device)
            gt_data = gt_data.unsqueeze(0).to(args.device)
            pred = model(inp_data)
            pred[pred < 0] = 0
            pred[pred > 1] = 1
            save_denoise_img(pred, inp_path[0])
            loss = loss_func(pred, gt_data)
            org_mse = loss_func(inp_data, gt_data)
            # batch size 1
            psnr = 10 * torch.log10(1/loss).item()
            org_psnr = 10 * torch.log10(1/org_mse).item()
            total_loss += loss.item()
            total_psnr += psnr
            total_org_psnr += org_psnr
            if batch_i % args.print_iters == 0:
                logger.info("Epoch: {} Batch: {}/{} | val_loss: {:.6f} | Mean loss: {:.6f}, psnr: {:.2f}, mean psnr: {:.2f}, org_psnr: {:.2f}, mean org_psnr: {:.2f}".format(
                    epoch, batch_i+1, len(dataloader), loss.item(), total_loss/(batch_i+1), psnr, total_psnr/(batch_i+1), org_psnr, total_org_psnr/(batch_i+1)))
            #if batch_i > 10: break
    logger.info('mean psnr: {}'.format(total_psnr / len(dataloader)))
    logger.info('mean org psnr: {}'.format(total_org_psnr / len(dataloader)))
    logger.info('mean loss: {}'.format(total_loss / len(dataloader)))
    return total_loss / len(dataloader)


def main():
    args = get_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    device = args.device

    logger = setup_logger("brdnet", args.save_dir, 0)
    logger.info("Using device {}".format(device))
    logger.info(args)

    val_data = CTDataset(data_path=args.val_path, patch_n=None, patch_size=None)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    model = BRDNet()
    model.load_state_dict(torch.load(args.pretrained))
    model.to(device)

    loss_func = nn.MSELoss()
    v_loss = val(val_loader, model, loss_func, 0, args, logger)

    logger.info('done')


if __name__ == '__main__':
    main()


