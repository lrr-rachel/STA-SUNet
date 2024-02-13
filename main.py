
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
import cv2
from torch.utils.data import DataLoader


gc.collect()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='Main code')
parser.add_argument('--root_distorted', type=str, default='../low_light_data_capture/input/', help='train and test datasets')
parser.add_argument('--root_restored', type=str, default='../low_light_data_capture/gt/', help='save output images')

parser.add_argument('--resultDir', type=str, default='STASUNet', help='save output images')
parser.add_argument('--unetdepth', type=int, default=5, metavar='N',  help='number of depths')
parser.add_argument('--numframes', type=int, default=5, metavar='N',  help='number of epochs to train (default: 10)')
parser.add_argument('--cropsize', type=int, default=512)
parser.add_argument('--savemodelname', type=str, default='model')
parser.add_argument('--maxepoch', type=int, default=10)
parser.add_argument('--NoNorm', action='store_false', help='Run test only')
parser.add_argument('--deform', action='store_true', help='Run test only')
parser.add_argument('--network', type=str, default='STASUNet')
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--recon_network', type=str, default='resnet', help='For pcd module')
parser.add_argument('--topleft', action='store_true', help='crop using top left')
parser.add_argument('--num_feat',type=int,default=16, help='features for pcd') 
parser.add_argument('--lowlightmode', action='store_true', help='using lowlight dataset')
parser.add_argument('--embed_dim',type=int,default=16, help='patch embedding dimension for sunet') 

parser.add_argument('--resize_height',type=int,default=512,help='resize image')
parser.add_argument('--resize_width',type=int,default=512, help='resize image')

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
maxepoch = args.maxepoch
NoNorm = args.NoNorm
deform = args.deform
network = args.network
retrain = args.retrain
recon_network = args.recon_network
topleft = args.topleft
num_feat = args.num_feat
lowlightmode = args.lowlightmode
embed_dim = args.embed_dim

if not os.path.exists(resultDir):
    os.mkdir(resultDir)

train_txt_file = '../low_light_data_capture/train_list.txt'
test_txt_file = '../low_light_data_capture/test_list.txt'
class LowLightDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root_distorted, root_restored='', network='STASUNet', numframes=3, transform=None):
        self.root_distorted = root_distorted
        self.root_restored = root_restored
        self.transform = transform

        # Read folder names from the training txt file
        with open(train_txt_file, 'r') as file:
            self.folder_names = file.read().splitlines()
        print("read training folder names: ", self.folder_names)
  

        self.filesnames = []  # Restored directory names
        self.distortednames = []  # Distorted directory names

        for folder_name in self.folder_names:
            data_folder_distorted = os.path.join(self.root_distorted, folder_name)
            data_folder_restored = os.path.join(self.root_restored, folder_name)

            self.filesnames.extend(glob.glob(os.path.join(data_folder_restored, 'normal_light_10', '*.png')))
            self.filesnames.extend(glob.glob(os.path.join(data_folder_restored, 'normal_light_20', '*.png')))
            self.distortednames.extend(glob.glob(os.path.join(data_folder_distorted, 'low_light_10', '*.png')))
            self.distortednames.extend(glob.glob(os.path.join(data_folder_distorted, 'low_light_20', '*.png')))

        print("Length restored = ", len(self.filesnames))
        print("Length distorted = ", len(self.distortednames))
        self.numframes = numframes


    def __len__(self):
        return len(self.filesnames)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        curf = idx+1

        halfcurf = int(self.numframes/2)
        totalframes = len(self.filesnames)
        
        if curf-halfcurf<=1:
            # close to the beginning of the sequence
            rangef = range(1,self.numframes+1)
        elif curf+halfcurf>=totalframes:
            # close to the end of the sequence
            if self.numframes==1:
                rangef = range(curf, curf+1)
            else:
                rangef = range(totalframes-self.numframes+1, totalframes+1)
        else:
            rangef = range(curf-halfcurf + 1 - (self.numframes % 2), curf+halfcurf+1)

        for f in rangef:
            rootdistorted = self.distortednames[f-1]
            # print('Read Distorted: '+rootdistorted)
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

        rootrestored = self.filesnames[idx]
        groundtruth = cv2.imread(rootrestored, cv2.IMREAD_COLOR)

        groundtruth = groundtruth.astype('float32')
        groundtruth = groundtruth/255.
        sample = {'image': image, 'groundtruth': groundtruth}
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomCrop(object):
    def __init__(self, output_size, topleft=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.topleft = topleft
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        if (h > new_h) and (not topleft):
            top = np.random.randint(0, h - new_h)
        else:
            top = 0
        if (w > new_w) and (not topleft):
            left = np.random.randint(0, w - new_w)
        else:
            left = 0
        image = image[top: top + new_h,
                      left: left + new_w]
        groundtruth = groundtruth[top: top + new_h,
                      left: left + new_w]
        return {'image': image, 'groundtruth': groundtruth}

class ToTensor(object):
    def __init__(self, network='unet'):
        self.network = network
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if network=='STASUNet':
            image = image.transpose((3, 2, 0, 1))
        else:
            image = image.transpose((2, 0, 1))
        groundtruth = groundtruth.transpose((2, 0, 1))
        image = torch.from_numpy(image.copy())
        groundtruth = torch.from_numpy(groundtruth.copy())
        # image
        if network=='STASUNet':
            image = (image-0.5)/0.5
        else:
            vallist = [0.5]*image.shape[0]
            normmid = transforms.Normalize(vallist, vallist)
            image = normmid(image)
        # ground truth
        vallist = [0.5]*groundtruth.shape[0]
        normmid = transforms.Normalize(vallist, vallist)
        groundtruth = normmid(groundtruth)
        image = image.unsqueeze(0)
        groundtruth = groundtruth.unsqueeze(0)
        return {'image': image, 'groundtruth': groundtruth}

class RandomFlip(object):
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        op = np.random.randint(0, 3)
        if op<2:
            image = np.flip(image,op)
            groundtruth = np.flip(groundtruth,op)
        return {'image': image, 'groundtruth': groundtruth}

# data loader
if lowlightmode:
    heathaze_dataset = LowLightDataset(root_distorted=root_distorted,
                                        root_restored=root_restored, network=network, numframes=numframes,
                                        transform=transforms.Compose([RandomCrop(cropsize, topleft=topleft),RandomFlip(),ToTensor(network=network)]))


# Check for the availability of GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Number of available GPUs: {torch.cuda.device_count()}")

# batch_size = 32
# data_loader = DataLoader(heathaze_dataset, batch_size=batch_size)

resultDirOutImg = os.path.join(resultDir,savemodelname)
if not os.path.exists(resultDirOutImg):
    os.mkdir(resultDirOutImg)

resultDirOut = os.path.join(resultDir,"trainingImg")
if not os.path.exists(resultDirOut):
    os.mkdir(resultDirOut)

if network=='STASUNet':
    model = STASUNet(num_in_ch=3,num_out_ch=3,num_feat=num_feat,num_frame=numframes,deformable_groups=8,num_extract_block=5,
                    num_reconstruct_block=10,center_frame_idx=None,hr_in=True,img_size=cropsize,patch_size=4,embed_dim=embed_dim, depths=[8, 8, 8, 8],num_heads=[8, 8, 8, 8],
                    window_size = 8,patch_norm=True,final_upsample="Dual up-sample")

if retrain:
    model.load_state_dict(torch.load(os.path.join(resultDir, 'best_'+savemodelname+'.pth.tar'),map_location=device))
model = model.to(device)
# use all GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.L1Loss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=0.000001)

# =====================================================================
print("[INFO] Start Training...")
# Log the time
time_start = time.time()
num_epochs=maxepoch
since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 100000000.0
train_loss = []
for epoch in range(num_epochs+1):
    for phase in ['train']:#, 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data
        
        # for batch in data_loader:
        #     inputs = batch['image'].to(device)
        #     labels = batch['groundtruth'].to(device)
        for i in range(len(heathaze_dataset)):
            sample = heathaze_dataset[i]
            inputs = sample['image'].to(device)
            labels = sample['groundtruth'].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
            # with torch.no_grad():
                outputs = model(inputs)
                if (i < 20):
                    output = outputs.clone()
                    output = output.squeeze(0)
                    output = output.detach().cpu().numpy()
                    output = output.transpose((1, 2, 0))
                    output = (output*0.5 + 0.5)*255
                    cv2.imwrite(os.path.join(resultDirOut, 'training'+ str(epoch) + '_'+str(i)+ '.png'), output.astype(np.uint8))

                loss = criterion(outputs, labels)
                #loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(heathaze_dataset)
        train_loss.append(epoch_loss)
            
        print('[Epoch] ' + str(epoch),':' + '[Loss] ' + str(epoch_loss))

        torch.save(model.state_dict(), os.path.join(resultDir, savemodelname + '_ep'+str(epoch)+'.pth.tar'))
        np.save(os.path.join(resultDir,'loss_array.npy'), np.array(train_loss))
        # deep copy the model
        if (epoch>=0) and (epoch_loss < best_acc):
            best_acc = epoch_loss
            torch.save(model.state_dict(), os.path.join(resultDir, 'best_'+savemodelname+'.pth.tar'))

# Log the time again
time_end = time.time()
print ("It took %d hours for training." % ((time_end-time_start)/3600))

