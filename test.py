from __future__ import division
import os
import time
import glob
import datetime
import argparse
import numpy as np
import tifffile as tiff
import cv2
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from arch_unet import UNet
from datalotest import datalo

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25")
parser.add_argument('--data_dir', type=str, default='./Imagenet_val')
parser.add_argument('--val_dirs', type=str, default='./validation')
parser.add_argument('--save_model_path', type=str, default='./results')
parser.add_argument('--log_name', type=str, default='unet_gauss25_b4e100r02')
parser.add_argument('--gpu_devices', default='1', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=1)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_snapshot', type=int, default=2)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--patchsize', type=int, default=256)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--increase_ratio", type=float, default=2.0)

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
network = UNet(in_nc=opt.n_channel+4,
               out_nc=opt.n_channel,
               n_feature=opt.n_feature)
if opt.parallel:
    network = torch.nn.DataParallel(network)
network = network.cuda()
network.load_state_dict(torch.load('./results'
                                   '/unet_gauss25_b4e100r02/2023-04-21-20-46/epoch_model_1000.pth'))

testdatalo=datalo('data2',batch_size=1,lengthofdata=1000,train=False)
np.random.seed(101)
n=1
numofch=1
with torch.no_grad():
    imgs = torch.zeros((1000, int(512/n), int(512/n)))
    data = torch.zeros((1000, int(512), int(512)))
    for j, (img) in enumerate(testdatalo):
        print(j)
        # img=img[:,0:1,:,:]
        img = img.cuda()
        img2 = img
        img = network(img)
        imgs[j, :, :] = img[:, numofch-1:numofch, :, :]
        data[j, :, :] = img2[:, 1:2, :, :]
    tiff.imsave('ch{}.tif'.format(numofch), np.float32((imgs).cpu().detach().numpy()))
    # tiff.imsave('denoise0003_max_numofmax=100_unregist.tif', np.float32((imgs).cpu().detach().numpy()))
    # tiff.imsave('noise0003.tif', np.float32(abs(imgs - data).cpu().detach().numpy()))
    # tiff.imsave('data0003.tif', np.float32((data).cpu().detach().numpy()))