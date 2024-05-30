import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from arch.arch import STASUNet
import time
import os
import glob
import cv2
import datasets
import yaml
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Main code')
parser.add_argument("--config", default='STASUNet.yml', type=str, help="training config file")
parser.add_argument('--resultDir', type=str, default='STASUNet', help='save output location')
parser.add_argument('--savemodelname', type=str, default='model')
parser.add_argument('--retrain', action='store_true')

parser.add_argument('--unetdepth', type=int, default=5, metavar='N',  help='PCDUNet depth')
parser.add_argument('--NoNorm', action='store_false', help='no normalisation layer')
args = parser.parse_args()

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

with open(args.config, "r") as f:
    config = yaml.safe_load(f)
config = dict2namespace(config)


if torch.cuda.is_available():
    print('using GPU')
else:
    print('using CPU')

if not os.path.exists(args.resultDir):
    os.mkdir(args.resultDir)

def train():
    dataset = datasets.__dict__[config.dataset.type](config)
    train_data, val_data = dataset.load_lowlight()

    # Check the availability of GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")

    resultDirModel = os.path.join(args.resultDir,args.savemodelname)
    if not os.path.exists(resultDirModel):
        os.mkdir(resultDirModel)

    resultDirOut = os.path.join(args.resultDir,"trainingImg")
    if not os.path.exists(resultDirOut):
        os.mkdir(resultDirOut)

    if config.model.network=='STASUNet':
        model = STASUNet(num_in_ch=config.model.num_in_ch,num_out_ch=config.model.num_out_ch,num_feat=config.model.num_feat,num_frame=config.dataset.num_frames,deformable_groups=config.model.deformable_groups,num_extract_block=config.model.num_extract_block,
                        num_reconstruct_block=config.model.num_reconstruct_block,center_frame_idx=None,hr_in=config.model.hr_in,img_size=config.dataset.image_size,patch_size=config.model.patch_size,embed_dim=config.model.embed_dim, depths=config.model.depths,num_heads=config.model.num_heads,
                        window_size = config.model.window_size,patch_norm=config.model.patch_norm,final_upsample="Dual up-sample")
    else:
        print("please specify model name!")

    if args.retrain:
        models = glob.glob(os.path.join(args.resultDirModel, args.savemodelname + '_ep*.pth.tar'))
        if len(models)>0:
            # get the last model
            epoch_start = max([int(os.path.basename(model).split('_ep')[1].split('.')[0]) for model in models])
            print("The latest epoch is: ", epoch_start)
        model.load_state_dict(torch.load(os.path.join(args.resultDirModel, args.savemodelname + '_ep'+str(epoch_start)+'.pth.tar'),map_location=device))
        epoch_start += 1 # only increase if further training

    model = model.to(device)
    # use all GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.L1Loss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)

    # =====================================================================
    # Log starting time
    time_start = time.time()
    num_epochs = config.training.maxepoch
    best_acc = 100000000.0
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs+1):
        for phase in ['train']:#, 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data = train_data
            else:
                model.eval()   # Set model to evaluate mode
                data = val_data
            running_loss = 0.0

            for i, sample in enumerate(tqdm(data, desc='Epoch ' + str(epoch))):
                # print(sample['image'].shape)
                # print(sample['groundtruth'].shape)
                inputs = sample['image'].to(device)
                labels = sample['groundtruth'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'): # track history if only in train
                    outputs = model(inputs)
                    if (i < 10): # load training samples
                        output = outputs.clone()
                        output = output.squeeze(0)
                        output = output.detach().cpu().numpy()
                        output = output.transpose((1, 2, 0))
                        output = (output*0.5 + 0.5)*255
                        cv2.imwrite(os.path.join(resultDirOut, 'training'+ str(epoch) + '_'+str(i)+ '.png'), output.astype(np.uint8))

                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_data)
            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)
                
            print('[Epoch] ' + str(epoch),':' + '[Loss] ' + str(epoch_loss))

            torch.save(model.state_dict(), os.path.join(resultDirModel, args.savemodelname + '_ep'+str(epoch)+'.pth.tar'))
            np.save(os.path.join(resultDirModel,'loss_array.npy'), np.array(train_loss))
            # deep copy the model
            if (epoch>2) and (epoch_loss < best_acc):
                best_acc = epoch_loss
                torch.save(model.state_dict(), os.path.join(args.resultDirModel, 'best_'+args.savemodelname+'.pth.tar'))

    # Log ending time 
    time_end = time.time()
    print ("Total %.2f hours for training." % ((time_end-time_start)/3600))

if __name__ == '__main__':
    train()