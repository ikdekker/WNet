# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:32:16 2018
@author: Tao Lin

Training and Predicting with the W-Net unsupervised segmentation architecture
"""

import os
import argparse
from datetime import datetime

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_enhance
import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision import datasets, transforms

import WNet

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with WNet')
parser.add_argument('--in_Chans', metavar='C', default=3, type=int, 
                    help='number of input channels')
parser.add_argument('--squeeze', metavar='K', default=4, type=int, 
                    help='Depth of squeeze layer')
parser.add_argument('--out_Chans', metavar='O', default=3, type=int, 
                    help='Output Channels')
parser.add_argument('--input_folder', metavar='f', default=None, type=str, 
                    help='Folder of input images')
parser.add_argument('--output_folder', metavar='of', default=None, type=str, 
                    help='folder of output images')
parser.add_argument('--load', metavar='of', default=None, type=str,
                    help='model')
parser.add_argument('--save_model', metavar='of', default=None, type=str,
                    help='directory to save model')


vertical_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,  0,  -1], 
                                            [1,  0,  -1], 
                                            [1,  0,  -1]]]])).float().cuda(), requires_grad=False)

horizontal_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,   1,  1], 
                                              [0,   0,  0], 
                                              [-1 ,-1, -1]]]])).float().cuda(), requires_grad=False)

def gradient_regularization(softmax, device='cuda'):
    vert=torch.cat([F.conv2d(softmax[:, i].unsqueeze(1), vertical_sobel) for i in range(softmax.shape[1])], 1)
    hori=torch.cat([F.conv2d(softmax[:, i].unsqueeze(1), horizontal_sobel) for i in range(softmax.shape[1])], 1)
    # print('vert', torch.sum(vert))
    # print('hori', torch.sum(hori))
    mag=torch.pow(torch.pow(vert, 2)+torch.pow(hori, 2), 0.5)
    mean=torch.mean(mag)
    return mean

def train_op(model, optimizer, input, psi=0.5):
    enc = model(input, returns='enc')

    n_cut_loss=gradient_regularization(enc)*psi
    n_cut_loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    dec = model(input, returns='dec')
    rec_loss=torch.mean(torch.pow(torch.pow(input, 2) + torch.pow(dec, 2), 0.5))*(1-psi)

    rec_loss.backward()
    # print(f"Loss: {rec_loss}")
    optimizer.step()

    optimizer.zero_grad()

    return (model, rec_loss)

def test():
    wnet=WNet.WNet(4)
    wnet=wnet.cuda()
    synthetic_data=torch.rand((1, 3, 128, 128)).cuda()
    optimizer=torch.optim.SGD(wnet.parameters(), 0.001)
    train_op(wnet, optimizer, synthetic_data)

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, labels in dl:
        print(len(images))
        show_images(images, nmax)
        break

def main():
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(f"dir_path: {dir_path}")

    image_size = (128, 128)
    transforms = T.Compose([
        T.Resize(image_size),
        T.ToTensor()
    ])

    # Download BSD500 dataset
    torch_enhance.datasets.BSDS500()

    dataset = datasets.ImageFolder("./data", transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    if not args.load:

        if not args.save_model:
            print("Provide a name for the model")
            return

        save_model_dir = f"models/{args.save_model}"
        os.makedirs(save_model_dir)

        wnet=WNet.WNet(4)
        wnet=wnet.cuda()

        learning_rate = 0.0003
        optimizer=torch.optim.SGD(wnet.parameters(), learning_rate)

        start_time = datetime.now()
        loss = 0
        for epoch in range(1, 50000):
            if epoch % 1000 == 0:
                learning_rate /= 10
                print(f"Reducing learning rate to {learning_rate}")
                optimizer=torch.optim.SGD(wnet.parameters(), learning_rate)

            if epoch % 100 == 0:
                print("==============================")
                print("Epoch = " + str(epoch))
                duration = (datetime.now() - start_time).seconds
                print(f"Loss: {loss}")
                print(f"Duration: {duration}s")

                model_name = f"{args.save_model}-{epoch}.pt"
                model_path = f"{save_model_dir}/{model_name}"
                print(f"Saving current model as '{model_path}'")
                torch.save(wnet, model_path)




                start_time = datetime.now()



            batch, labels = next(iter(dataloader))
            batch = batch.cuda()
            wnet, loss = train_op(wnet, optimizer, batch)


            # duration = (datetime.now() - start_time).seconds
            # print(f"Duration: {duration}s")

        torch.save(wnet, "models/75.pt")
    else:
        wnet = torch.load(args.load, map_location='cpu')

        encodings = []
        for i, (batch, labels) in enumerate(dataloader):
            if i == 50:
                break

            print(f"\rSegmenting image {i}/{len(dataloader)}", end='')
            # batch = batch.cuda()
            enc = wnet.forward(batch, returns="enc")
            encodings.append(enc[0].detach().numpy())

        encodings = torch.tensor(encodings)
        plt.imshow(torchvision.utils.make_grid(encodings, nrow=10).permute(1, 2, 0))
        plt.show()




if __name__ == "__main__":
    main()
