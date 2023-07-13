import os

import random


import tifffile as tiff
import numpy as np
import torch
import torch.utils.data as Data
import scipy.io as io
# import matplotlib.pyplot as pl
from torchvision import transforms

from torch import zeros
# lenth=1000

# image_size = 512
# channels = 1
# batch_size = 4
#
# image_size_crop = 512

# p1 = random.randint(0, 1)
# p2 = random.randint(0, 1)
# p3=random.randint(0, 180)
# im_aug = transforms.Compose([
#     transforms.RandomHorizontalFlip(p1),
#     transforms.RandomVerticalFlip(p2),
#     # transforms.RandomRotation(180, resample=False, expand=False, center=None),
#     transforms.Resize((imgl, imgl)),
#     # transforms.RandomRotation(p3, resample=False, expand=False, center=None)
#
# ])



class TensorsDataset(torch.utils.data.Dataset):
    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''

    def __init__(self, data_tensor,data_max,train=True):

        self.data_tensor = data_tensor
        self.data_max= data_max

        self.train=train







        # self.transforms = im_aug


    def __getitem__(self, index):
        # seed = np.random.randint(2147483647)
        # random.seed(seed)
        # torch.manual_seed(seed)
        # if self.train:
        #     p1 = random.randint(0, 1)
        #     p2 = random.randint(0, 1)
        # else:
        #     p1=0
        #     p2=0

        im_aug = transforms.Compose([
            # transforms.Resize(image_size),
            # transforms.RandomCrop(256)
            # transforms.CenterCrop(image_size_crop),
            # transforms.Lambda(lambda t: (t * 2) - 1)  # 此处将输入数据从(0,1)区间转换到(-1,1)区间

        ])

        data_tensor = self.data_tensor[index,:,:,:]
        max_tensor= self.data_max[index,:,:,:]
        t = torch.cat((data_tensor, max_tensor), dim=0)
        # print(t.shape)


        # data_tensor = im_aug(data_tensor)


        return t

    def __len__(self):
        return self.data_tensor.size(0)
def datalo(im_folder = 'data',train=True,batch_size = 4,lengthofdata=1000):
    noise_im_all=torch.zeros(0,1,512,512)
    noise_max_all = torch.zeros(0, 1, 512, 512)
    numberofmax=100
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name

        noise_im = tiff.imread(im_dir)[0:lengthofdata,:,:]
        # noise_immean = np.expand_dims(noise_im.mean(0), 0).repeat(lenth, axis=0)

        print('noise_im shape -----> ',noise_im.shape)
        print('noise_im max -----> ',noise_im.max())
        print('noise_im min -----> ',noise_im.min())

        noise_im=(noise_im-noise_im.min())/(noise_im.max()-noise_im.min())
        noise_im = noise_im.astype(np.float32)
        noise_im = torch.tensor(noise_im)


        trans = transforms.Compose([
            # transforms.Resize(image_size),
            transforms.RandomAffine(degrees=0.5,translate=(0.0,0.005)),
            transforms.RandomCrop(512),
            # transforms.CenterCrop(512),
            # transforms.Lambda(lambda t: (t * 2) - 1)  # 此处将输入数据从(0,1)区间转换到(-1,1)区间

        ])
        # noise_im2=torch.zeros(lengthofdata,512,512)
        # for i,noise_im_one in enumerate(noise_im):
        #     noise_im2[i,:,:]=trans(noise_im[i:i+1,:,:])

        # noise_im2=noise_im
        # noise_max,_=noise_im.max(0)
        # noise_max= noise_im.mean(0)
        # noise_max=noise_max.repeat(lengthofdata,1,1)
        noise_max =torch.zeros(0,512,512)
        for i in range(int(lengthofdata/numberofmax)):
            # tempmax,_=noise_im[i*numberofmax:(i+1)*numberofmax,:,:].max(0)

            tempmax = noise_im[i * numberofmax:(i + 1) * numberofmax, :, :].mean(0)
            # tiff.imsave('./imagemax/'+im_name+'+{}+mean.tif'.format(i), np.float32((tempmax).numpy()))
            tempmax=tempmax.repeat(numberofmax,1,1)
            noise_max=torch.cat((noise_max, tempmax), dim=0)





        noise_im=noise_im.unsqueeze(1)
        noise_max=noise_max.unsqueeze(1)
        noise_im_all=torch.cat((noise_im_all,noise_im), dim=0)
        noise_max_all = torch.cat((noise_max_all, noise_max), dim=0)






        # torch_dataset = Data.TensorDataset(data_tensor=x , target_tensor = y )
    print(noise_im_all.shape)
    print(noise_max_all.shape)
    torch_dataset =TensorsDataset(noise_im_all,noise_max_all,train=train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True
    )
    # print(loader)
    #
    # for epoch in range(1):
    #     for step, (batch_x, batch_y) in enumerate(loader):
    #         print('Epoch:', epoch, '| step:', step, '|batch_x:', batch_x.size(), '|batch_y：', batch_y.size())

    return loader
# testdatalo=datalo('datatest',batch_size=1,lengthofdata=3000,train=False)
# # datalo('data')
# for i ,img in enumerate(testdatalo):
#     print(i)
#     pass
