from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from arch import PCDAlignment, STASUNet
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import glob
from torch.utils.data import Dataset
from skimage import io, transform
import shutil
from PIL import Image
import gc
import matplotlib.pyplot as plt
from math import log10, sqrt
import cv2
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio 
from skimage.metrics import structural_similarity 
from scipy.stats import norm

parser = argparse.ArgumentParser(description='Testing')

parser.add_argument('--root_distorted', type=str, default='../low_light_data_capture/input/', help='train and test datasets')
parser.add_argument('--root_restored', type=str, default='../low_light_data_capture/gt/', help='save output images')
parser.add_argument('--root_test', type=str, default='../low_light_data_capture/input/', help='save output images')

parser.add_argument('--resultDir', type=str, default='STASUNet', help='save output images')
parser.add_argument('--unetdepth', type=int, default=5, metavar='N',  help='number of epochs to train (default: 10)')
parser.add_argument('--numframes', type=int, default=5, metavar='N',  help='number of epochs to train (default: 10)')
parser.add_argument('--cropsize', type=int, default=512)
parser.add_argument('--savemodelname', type=str, default='model')
parser.add_argument('--NoNorm', action='store_false', help='Run test only')
parser.add_argument('--network', type=str, default='STASUNet')
parser.add_argument('--topleft', action='store_true', help='crop using top left')
parser.add_argument('--num_feat',type=int,default=16, help='features for pcd') 
parser.add_argument('--embed_dim',type=int,default=16, help='patch embedding dimension for sunet')

args = parser.parse_args()

if torch.cuda.is_available():
    print('using GPU')
else:
    print('using CPU')

root_distorted = args.root_distorted
root_restored  = args.root_restored
resultDir = args.resultDir
unetdepth = args.unetdepth
numframes = args.numframes
cropsize = args.cropsize
savemodelname = args.savemodelname
NoNorm = args.NoNorm
network = args.network
topleft = args.topleft
num_feat = args.num_feat
root_test = args.root_test
embed_dim = args.embed_dim

test_txt_file = '../low_light_data_capture/test_list.txt'
output_file = 'test_' + resultDir + '.txt'

resultDirOutImg = os.path.join(resultDir,savemodelname)
if not os.path.exists(resultDirOutImg):
    os.mkdir(resultDirOutImg)
if network=='STASUNet':
    model = STASUNet(num_in_ch=3,num_out_ch=3,num_feat=num_feat,num_frame=numframes,deformable_groups=8,num_extract_block=5,
                num_reconstruct_block=10,center_frame_idx=None,hr_in=True,img_size=cropsize,patch_size=4,embed_dim=embed_dim, depths=[8, 8, 8, 8],num_heads=[8, 8, 8, 8],
                window_size = 8,patch_norm=True,final_upsample="Dual up-sample")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model.load_state_dict(torch.load(os.path.join(resultDir,'best_model.pth.tar'),map_location=device))
except:
    checkpoint = torch.load(os.path.join(resultDir, 'best_model.pth.tar'), map_location=device)
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key[7:]] = checkpoint[key]  # Remove 'module.' prefix
            del checkpoint[key]
    model.load_state_dict(checkpoint)

model = model.to(device)


# crop input size
overlapRatio = 1./4.
hpatch = 512
wpatch = 512
hgap = int(float(hpatch)*overlapRatio)
wgap = int(float(wpatch)*overlapRatio)

# Create weight for each patch
a = norm(hpatch/2, hpatch/6).pdf(np.arange(hpatch)) # gaussian weights along width
b = norm(wpatch/2, wpatch/6).pdf(np.arange(wpatch)) # gaussian weights along height
wmap = np.matmul(a[np.newaxis].T,b[np.newaxis]) # 2D weight map
wmap = wmap/wmap.sum()

# Repeat the 2D weight map along the third dimension
wmap = np.repeat(wmap[:, :, np.newaxis], 3, axis=2)
# print("wmap",wmap)

# =====================================================================

# Read folder names from the testing txt file
with open(test_txt_file, 'r') as file:
    testfolder_names = file.read().splitlines()
# print("read testing folder names: ", testfolder_names)

filesnames = []  # Restored directory names
distortednames = []  # Distorted directory names

for folder_name in testfolder_names:
    data_folder_distorted = os.path.join(root_distorted, folder_name)
    data_folder_restored = os.path.join(root_restored, folder_name)

    filesnames.extend(glob.glob(os.path.join(data_folder_restored, 'normal_light_10', '*.png')))
    filesnames.extend(glob.glob(os.path.join(data_folder_restored, 'normal_light_20', '*.png')))

    distortednames.extend(glob.glob(os.path.join(data_folder_distorted, 'low_light_10', '*.png')))
    distortednames.extend(glob.glob(os.path.join(data_folder_distorted, 'low_light_20', '*.png')))


print("Number of test images:", len(distortednames))
list_psnr = []
list_ssim = []
num_saved_images_per_scene_light10 = {} 
num_saved_images_per_scene_light20 = {}
num_saved_images_per_scene = {}

# Keep track of the current scene
current_scene = None
scene_psnr_values = []
scene_ssim_values = []

for i in range(len(filesnames)):
    subname = filesnames[i].split("/")
    subdist = os.path.split(distortednames[i])[0]
    lightlevel = os.path.split(subdist)[-1]
    scenename = os.path.split(subdist)[0].split('/')[-1]

    curf = i+1

    halfcurf = int(numframes/2)
    totalframes = len(filesnames)
    
    if curf-halfcurf<=1:
        # close to the beginning of the sequence
        rangef = range(1,numframes+1)
    elif curf+halfcurf>=totalframes:
        # close to the end of the sequence
        if numframes==1:
            rangef = range(curf, curf+1)
        else:
            rangef = range(totalframes-numframes+1, totalframes+1)
    else:
        rangef = range(curf-halfcurf + 1 - (numframes % 2), curf+halfcurf+1)

    for f in rangef:
        rootdistorted = distortednames[f-1]
        temp = cv2.imread(rootdistorted, cv2.IMREAD_COLOR)
        temp = temp.astype('float32')

        if network=='STASUNet':
            temp = temp[..., np.newaxis]
        if f==rangef[0]:
            image = temp/255.
        else:
            if network=='STASUNet':
                image = np.append(image,temp/255.,axis=3)
            else:
                image = np.append(image,temp/255.,axis=2)

    img = image
    # Run through each overlaping patch
    himg = img.shape[0]
    wimg = img.shape[1]

    # weightMap = np.zeros((himg,wimg,3),np.float32) + 0.00001
    weightMap = np.zeros((himg,wimg,3),np.float32)
    probMap = np.zeros((himg,wimg,3),np.float32)

    for starty in np.concatenate((np.arange( 0, himg-hpatch, hgap),np.array([himg-hpatch])),axis=0):
        for startx in np.concatenate((np.arange( 0, wimg-wpatch, wgap),np.array([wimg-wpatch])),axis=0):
            crop_img = img[starty:starty+hpatch, startx:startx+wpatch]

            weightMap[starty:starty+hpatch, startx:startx+wpatch] += wmap
                
            # Reshape as needed to feed into model
            image = crop_img
            if network=='STASUNet':
                image = image.transpose((3, 2, 0, 1))
            else:
                image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)
            if network=='STASUNet':
                image = (image-0.5)/0.5
            else:
                vallist = [0.5]*image.shape[0]
                normmid = transforms.Normalize(vallist, vallist)
                image = normmid(image)
            image = image.unsqueeze(0)

            inputs = image.to(device)
            with torch.no_grad():
                output = model(inputs)
                output = output.squeeze(0)
                output = output.cpu().numpy()
                output = output.transpose((1, 2, 0))

            probMap[starty:starty+hpatch, startx:startx+wpatch] += output*wmap
    # print("weightMap",weightMap)
    # Normalised weight
    probMap /= weightMap

    # clip to range [-1,1]
    probMap = np.clip(probMap, -1, 1)

    probMap = (probMap*0.5 + 0.5)*255


    # save the first 30 output frames for each scene    
    if int(subname[-1].split('.')[0]) < 30:
	    cv2.imwrite(os.path.join(resultDirOutImg, scenename+'_'+lightlevel+'_'+subname[-1]),probMap.astype(np.uint8))


    pred = probMap
    gt = cv2.imread(filesnames[i], cv2.IMREAD_COLOR)

    gt = gt.astype('float32')

    psnrvalue = peak_signal_noise_ratio (gt, pred, data_range=255)
    ssimvalue = structural_similarity(gt, pred, channel_axis=2, data_range=255, multichannel=True)

    list_psnr.append(psnrvalue)
    list_ssim.append(ssimvalue)

# Calculate and print average PSNR and SSIM
print('Average PSNR:', np.mean(list_psnr))
print('Average SSIM:', np.mean(list_ssim))

# Save the values to a text file
with open(output_file, 'a') as file:
    file.write(f'Total Average PSNR: {np.mean(list_psnr)}\n')
    file.write(f'Total Average SSIM: {np.mean(list_ssim)}\n')

print(f'Values saved to {output_file}')


   

