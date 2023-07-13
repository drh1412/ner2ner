import torch.multiprocessing as mp
from time import sleep

import torch

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
sizeofstack=401



class p1filter():

    def __init__(self):


        self.freelength=torch.tensor([sizeofstack]).share_memory_()

        self.t1=torch.zeros(sizeofstack,1,512,512).share_memory_()
        self.t2 = torch.zeros(sizeofstack, 1, 512, 512).share_memory_()
        self.t3 = torch.zeros(sizeofstack, 1, 512, 512).share_memory_()
        self.t1start=torch.tensor([0]).share_memory_()
        self.t1process=torch.tensor([0]).share_memory_()
        self.t1end=torch.tensor([0]).share_memory_()
        self.t2start=torch.tensor([0]).share_memory_()
        self.t2process=torch.tensor([0]).share_memory_()
        self.t2end=torch.tensor([0]).share_memory_()
        self.t3start=torch.tensor([0]).share_memory_()
        self.t3process=torch.tensor([0]).share_memory_()
        self.t3end=torch.tensor([0]).share_memory_()


        # p1 = mp.Process(target=self.removebg)
        # p2 = mp.Process(target=self.fun, args=(self.freelength,2))
        # p3 = mp.Process(target=self.fun, args=(self.freelength, 2))
        # p1.start()
        # p2.start()
        # p3.start()


        # while True:
        #     sleep(0.5)
        #     print(self.t1end)

    def readdata(self,data,length):
        # print(id(self.t1end))
        # print(self.getfreelength())
        # print(self.t1start)
        # print(self.t1end)
        # print(self.t1process)
        t1start=int(self.t1start[0])
        t1end=int(self.t1end[0])

        if t1end + length<sizeofstack:
            # print("t1end{}".format(t1end))
            self.t1[t1end:t1end+length,:,:,:]=data[:,:,:,:]
            # print("readdata")
            self.t1end[0]=t1end + length
            # self.freelength[0]=sizeofstack-self.t1end[0]
            # sleep(1)

            # print(self.t1end)

        else:
            if (length+t1end-sizeofstack)<t1start:
                self.t1[t1end:sizeofstack, :, :, :] = data[0:(sizeofstack-t1end)]
                self.t1[0:(length+t1end-sizeofstack), :, :, :] = data[(sizeofstack - t1end):length]
                self.t1end[0] = length+t1end-sizeofstack
                # self.freelength[0]=t1start-t1end
                # sleep(1)
                # print("readdata")
                # print(self.t1end)

            else:
                print("check length!")


    def getfreelength(self):
        t1start=int(self.t1start[0])
        t1end=int(self.t1end[0])
        if t1start<=t1end:
            freelength=sizeofstack-t1end+t1start
        else:
            freelength=t1start-t1end
        return freelength

    def model(self,data):
        with torch.no_grad():
            print("model")
            os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
            network = UNet(in_nc=opt.n_channel,
                                out_nc=opt.n_channel,
                                n_feature=opt.n_feature)

            # network =network.cuda()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            network =network.to(device)
            print(device)
            # self.network.load_state_dict(torch.load('./epoch_model_100.pth'))
            network.load_state_dict(
                torch.load('./results/unet_gauss25_b4e100r02/2023-04-02-15-31/epoch_model_050.pth'))
            print("load sucess")
            data=data.to(device)
            # try:
            #     data=network(data)
            #     print(data.shape)
            # except Exception as err:
            #     print('An exception happened: ' + str(err))

            # image=torch.zeros((100,1,512,512)).to(device)
            for i in range(100):
            #     print(data[i:i+1,:,:,:].shape)
            #     print(image[i:i+1,:,:,:].shape)
                data[i:i+1,:,:,:]=data[i:i+1,:,:,:]*10
                    # network(data[i:i+1,:,:,:])
            #     print(i)
            return data





    def removebg(self):
        with torch.no_grad():
            print("remove begin")
            print("model")
            os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
            network = UNet(in_nc=opt.n_channel,
                           out_nc=opt.n_channel,
                           n_feature=opt.n_feature)

            # network =network.cuda()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            network = network.to(device)
            print(device)
            # self.network.load_state_dict(torch.load('./epoch_model_100.pth'))
            network.load_state_dict(
                torch.load('./results/unet_gauss25_b4e100r02/2023-04-02-15-31/epoch_model_050.pth'))
            print("load sucess")


            while True:

                t1end= int(self.t1end[0])
                t1process=int(self.t1process[0])

                # print(id(self.t1end))
                if t1process!=t1end:
                    print("self.t1process")
                    print(t1end)
                    print(t1process)
                    try:

                        if t1process<t1end:

                            for i in range(t1process,t1end):
                                self.t1[i:i+1,:,:,:]=network( self.t1[i:i+1,:,:,:].to(device))

                            # data=self.model(self.t1[t1process:t1end,:,:,:])
                            # self.t1[t1process:t1end,:,:,:]=data

                        else:
                            for i in range(t1process,sizeofstack):
                                self.t1[i:i+1,:,:,:]=network( self.t1[i:i+1,:,:,:].to(device))
                            for i in range(0,t1end):
                                self.t1[i:i + 1, :, :, :] = network(self.t1[i:i + 1, :, :, :].to(device))


                            # self.t1[t1process:sizeofstack,:,:,:] = self.model(self.t1[t1process:sizeofstack,:,:,:])
                            # self.t1[0:t1end, :, :, :] = self.model(self.t1[0:t1end, :, :, :])
                        self.t1process[0]=t1end
                    except Exception as err:
                        print('An exception happened: ' + str(err))
                else:
                    pass
                    # sleep(1)
                    # print("ok")
                    # print(self.t1process)
                    # print(self.t1end)
    def getdata(self,length):
        data=torch.zeros((length,1,512,512))
        while True:
            t1end = int(self.t1end[0])
            t1process = int(self.t1process[0])
            t1start=int(self.t1start[0])
            # print(t1start)
            # print(t1process)
            # sleep(0.5)

            if t1start<t1process:
                    print(t1start)
                    print(t1process)
                    if (t1process-t1start)>=length:
                        data[:,:,:,:]=self.t1[t1start:t1start+length,:,:,:]
                        self.t1start[0]=t1start+length
                        print("return")
                        return data
                    else :
                        pass
            elif t1start>t1process:
                print(t1start)
                print(t1process)
                if(sizeofstack-t1start+t1process)>=length:
                    if sizeofstack-t1start>=length:
                        data[:, :, :, :] = self.t1[t1start:t1start + length, :, :, :]
                        self.t1start[0] = t1start + length
                        print("return")
                        return data
                    else:
                        data[0:(sizeofstack-t1start),:,:,:]=self.t1[t1start:sizeofstack,:,:,:]
                        data[(sizeofstack-t1start):length,:,:,:]=self.t1[0:(length+t1start-sizeofstack),:,:,:]
                        self.t1start[0] = length+t1start-sizeofstack
                        print("return")
                        return data

            else:
                    pass

def test(f):
    print("get begin")

    with torch.no_grad():
        imgs = torch.zeros((1000, int(512 ), int(512 )))
        i=0
        while i<10:
                print("get:{}".format(i))
                imgs[i*100:(i+1)*100,:,:]=f.getdata(100).squeeze()
                # print(imgs[i*100:(i+1)*100,:,:].shape)
                # print(f.getdata(100).squeeze().shape)

                i=i+1




        tiff.imsave('testthread.tif', np.float32((imgs).cpu().detach().numpy()))


def test3(f):
    print(333333)
    while True:

        # a=f.getdata(100)
        print("test freelength{}".format(f.getfreelength()))
        print("test start{}".format(f.t1start))
        print("test end{}".format(f.t1end))
        print("test process{}".format(f.t1process))
        print("\n")
        sleep(2)
def test2(f):
    print(2)
    noise_im = tiff.imread('./data/img0001.tif')[0:1000, :, :]
    # noise_immean = np.expand_dims(noise_im.mean(0), 0).repeat(lenth, axis=0)

    noise_im = noise_im.astype(np.float32)
    noise_im = torch.tensor(noise_im)

    noise_im = (noise_im - noise_im.min()) / (noise_im.max() - noise_im.min())


    noise_im = noise_im.unsqueeze(1)
    i=0
    print('noise_im shape -----> ', noise_im.shape)
    print('noise_im max -----> ', noise_im.max())
    print('noise_im min -----> ', noise_im.min())
    while i<10:

            # print(f.getfreelength())
            if (f.getfreelength()>=100):

                # print("read")
                # data=torch.zeros((100,1,512,512))


                # data[:,:,:,:]=noise_im[i:(i+1)*100,:,:,:].clone()
                f.readdata(noise_im[i*100:(i+1)*100,:,:,:],100)

                i=i+1
                # print(i)


f=p1filter()
# p=mp.Process(target=test,args=tuple([f]))
# p.start()
# p2=mp.Process(target=test2,args=tuple([f]))
# p2.start()
# p3=mp.Process(target=test3,args=tuple([f]))
# p3.start()
if __name__ == '__main__':
    with torch.no_grad():
        ctx = torch.multiprocessing.get_context("spawn")
        print(torch.multiprocessing.cpu_count())
        pool = ctx.Pool(5)  # 7.7G
        # pool.apply_async(test3, tuple([f]))

        pool.apply_async(test, tuple([f]))
        pool.apply_async(test2, tuple([f]))
        pool.apply_async(f.removebg)
        pool.close()
        pool.join()
        # pool.join()



# data=torch.zeros(100,1,512,512)
# f.readdata(data,100)
# # a=f.getdata(100)
# f.readdata(data,100)
# f.readdata(data,100)
# f.readdata(data,100)
# data2=torch.zeros(1000,1,512,512)
# f.readdata(data2,1000)
# f.readdata(data2,1000)
# f.readdata(data2,1000)
# f.readdata(data,100)
# f.readdata(data,100)
# f.readdata(data,100)
# print(f.t1)
# sleep(10)
# print(f.t1)
# print(f.freelength)
# print(f.t1start)
# print(f.t1end)
# print(f.t1process)



# import torch.multiprocessing as mp
# import torch
#
# def foo(worker,tl):
#     tl[worker] += (worker+1) * 1000

# if __name__ == '__main__':
#     tl = [torch.randn(2), torch.randn(3)]
#
#     for t in tl:
#         t.share_memory_()
#
#     print("before mp: tl=")
#     print(tl)
#
#     p0 = mp.Process(target=foo, args=(0, tl))
#     p1 = mp.Process(target=foo, args=(1, tl))
#     p0.start()
#     p1.start()
#     p0.join()
#     p1.join()
#
#     print("after mp: tl=")
#     print(tl)

#coding=utf-8
from multiprocessing import Pool
from threading import Thread





