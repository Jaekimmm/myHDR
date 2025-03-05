import os
import random
import numpy as np
import torch
import h5py
import time

import torch.nn as nn
from torch.nn import init
import torchvision as tv
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

import glob
from utils import *
import imageio
from datetime import datetime
import wandb
from tqdm import tqdm


def model_restore(model, trained_model_dir):
    model_list = glob.glob((trained_model_dir + "/trained_*.pkl"))
    a = []
    for i in range(len(model_list)):
        index = int(model_list[i].split('model')[-1].split('.')[0])
        a.append(index)
    if len(a) == 0:
        return model, 0
    else:
        epoch = np.sort(a)[-1]
        model_path = trained_model_dir + 'trained_model{}.pkl'.format(epoch)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model, epoch


class data_loader(data.Dataset):
    def __init__(self, data_dir, crop_size=0, geometry_aug=False):
        super().__init__()
        self.crop_size = crop_size
        self.geometry_aug = geometry_aug
        
        self.data_name = data_dir.split("_")
        if len(self.data_name) > 1:
            self.data_name = self.data_name[1]
        else:
            raise ValueError("Invalid dataset path format. Expected format: dataset_{name}_{test/train}.")
        
        self.sample_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.length = len(self.sample_list)
        
        print(f"[INFO] (data_loader) {self.length} data loaded from {data_dir}")

    def __getitem__(self, index):
        sample_path = self.sample_list[index]
        sample_path = sample_path.strip()

        if os.path.exists(sample_path):
            #TODO : Add dataset here
            if self.data_name == 'sice':
                with h5py.File(sample_path, 'r') as f:
                    data1 = f['IN'][0:3, :, :]  # short
                    data2 = f['IN'][0:3, :, :]  # mid
                    data3 = f['IN'][3:6, :, :]  # long
                    label = f['GT'][:]
            elif self.data_name == 'kalan':
                with h5py.File(sample_path, 'r') as f:
                    data1 = f['IN'][ 3: 6, 0:1496, :]   # short after gain adjustment
                    data2 = f['IN'][ 9:12, 0:1496, :]  # mid after gain adjustment
                    data3 = f['IN'][15:18, 0:1496, :] # long after gain adjustment
                    label = f['GT'][  :  , 0:1496, :]
            
            if self.crop_size > 0 : 
                data1, data2, data3, label = self.imageCrop(data1, data2, data3, label, self.crop_size)
            if self.geometry_aug : 
                data1, data2, data3, label = self.image_Geometry_Aug(data1, data2, data3, label)
            
            data1 = torch.from_numpy(data1).float()
            data2 = torch.from_numpy(data2).float()
            data3 = torch.from_numpy(data3).float()
            label = torch.from_numpy(label).float()
                
        return data1, data2, data3 , label

    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)

    def imageCrop(self, data1, data2, data3, label, crop_size):
        c, w, h = data1.shape
        w_boder = w - crop_size  # sample point y
        h_boder = h - crop_size  # sample point x ...
        
        # if crop_size is larger than the image size, then do not crop
        if w_boder > 0:
            start_w = self.random_number(w_boder - 1)
            end_w = start_w + crop_size
        else:
            start_w = 0
            end_w = w
            
        if h_boder > 0:
            start_h = self.random_number(h_boder - 1)
            end_h = start_h + crop_size
        else:
            start_h = 0
            end_h = h

        crop_data1 =   data1[:, start_w:end_w, start_h:end_h]
        crop_data2 =   data2[:, start_w:end_w, start_h:end_h]
        crop_data3 =   data3[:, start_w:end_w, start_h:end_h]
        crop_label = label[:, start_w:end_w, start_h:end_h]
        return crop_data1, crop_data2, crop_data3, crop_label

    def image_Geometry_Aug(self, data1, data2, data3, label):
        c, w, h = data1.shape
        num = self.random_number(4)

        if num == 1:  # no aug
            in_data1 = data1
            in_data2 = data2
            in_data3 = data3
            in_label = label

        if num == 2:  # flip_left_right
            index = np.arange(w, 0, -1) - 1
            in_data1 = data1[:, index, :]
            in_data2 = data2[:, index, :]
            in_data3 = data3[:, index, :]
            in_label = label[:, index, :]

        if num == 3:  # flip_up_down
            index = np.arange(h, 0, -1) - 1
            in_data1 = data1[:, :, index]
            in_data2 = data2[:, :, index]
            in_data3 = data3[:, :, index]
            in_label = label[:, :, index]

        if num == 4:  # rotate 180
            index = np.arange(w, 0, -1) - 1
            in_data1 = data1[:, index, :]
            in_data2 = data2[:, index, :]
            in_data3 = data3[:, index, :]
            in_label = label[:, index, :]
            index = np.arange(h, 0, -1) - 1
            in_data1 = in_data1[:, :, index]
            in_data2 = in_data2[:, :, index]
            in_data3 = in_data3[:, :, index]
            in_label = in_label[:, :, index]

        return in_data1, in_data2, in_data3, in_label
        

def get_lr(epoch, lr, max_epochs):
    #if epoch <= max_epochs * 0.8:
    #    lr = lr
    #else:
    #    lr = 0.1 * lr
    return lr

def train(epoch, model, loss_model, train_loaders, optimizer, trained_model_dir, wandb, args):
    # Adjust the learning rate (TODO : not used for now ...)
    #lr = get_lr(epoch, args.lr, args.epochs)
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = lr
    #print('[INFO] lr: {}'.format(optimizer.param_groups[0]['lr']))
    
    model.train()
    loss_model.eval()
    
    num = 0
    trainloss = 0
    avg_loss = 0
    
    train_loader_iter = tqdm(enumerate(train_loaders), total=len(train_loaders), desc=f"Epoch {epoch}")
    for batch_idx, (data1, data2, data3, target) in train_loader_iter:
        if args.use_cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
            target = target.cuda()

        data1 = tonemap(data1, args.input_tonemap)
        data2 = tonemap(data2, args.input_tonemap)
        data3 = tonemap(data3, args.input_tonemap)
        target = tonemap(target, args.label_tonemap)
        
        #if (epoch == 1) and (batch_idx == 0):
        #    # print one sample data to check the image
        #    img = torch.squeeze(data1[0:1, :, :, :]*255.)
        #    img = img.data.cpu().numpy().astype(np.uint8)
        #    img = np.transpose(img, (2, 1, 0))
        #    img = img[:, :, [0, 1, 2]]
        #    imageio.imwrite(trained_model_dir + f'/sample_in1_b{batch_idx}.jpg', img)
        #    
        #    img = torch.squeeze(data2[0:1, :, :, :]*255.)
        #    img = img.data.cpu().numpy().astype(np.uint8)
        #    img = np.transpose(img, (2, 1, 0))
        #    img = img[:, :, [0, 1, 2]]
        #    imageio.imwrite(trained_model_dir + f'/sample_in2_b{batch_idx}.jpg', img)
        
        #    img = torch.squeeze(data3[0:1, :, :, :]*255.)
        #    img = img.data.cpu().numpy().astype(np.uint8)
        #    img = np.transpose(img, (2, 1, 0))
        #    img = img[:, :, [0, 1, 2]]
        #    imageio.imwrite(trained_model_dir + f'/sample_in3_b{batch_idx}.jpg', img)
        
        #    img = torch.squeeze(target[0:1, :, :, :]*255.)
        #    img = img.data.cpu().numpy().astype(np.uint8)
        #    img = np.transpose(img, (2, 1, 0))
        #    img = img[:, :, [0, 1, 2]]
        #    imageio.imwrite(trained_model_dir + f'/sample_targeti_b{batch_idx}.jpg', img)
        
        if args.offset: 
            # if output activation is tanh, then input should be normalized to [-1, 1]
            data1 = (data1 * 2.0) - 1.0
            data2 = (data2 * 2.0) - 1.0
            data3 = (data3 * 2.0) - 1.0
            target = (target * 2.0) - 1.0
            
        optimizer.zero_grad()
        output = model(data1, data2, data3)

        if args.offset: 
            # if output activation is tanh, then input should be normalized to [-1, 1]
            output = (output + 1.0) / 2.0
            target = (target + 1.0) / 2.0
        
        output = tonemap(output, args.output_tonemap)
                
        # Loss calculation
        if args.loss == 'vgg':
            output_vgg = loss_model(output)
            target_vgg = loss_model(target)
            
            loss_mse = F.mse_loss(output, target)
            loss_vgg = F.l1_loss(output_vgg, target_vgg)
            loss = loss_vgg + loss_mse
        else:
            loss = F.l1_loss(output, target)
        
        loss.backward()
        optimizer.step()
        trainloss = trainloss + loss
        avg_loss = avg_loss + loss
        
        train_loader_iter.set_postfix(loss=loss.item())
        
        if (batch_idx+1) % 10 == 0:
            trainloss = trainloss / 10
            wandb.log({
                "epoch": epoch,
                "batch": batch_idx + 1,
                "loss": trainloss.item(),
            })
            
            trainloss = 0

    avg_loss /= len(train_loaders)
    return avg_loss

def testing_fun(model, test_loaders, outdir, args):
    model.eval()
    
    test_loss = 0
    val_psnr = 0
    val_psnr_norm = 0
    val_psnr_mu = 0
    num = 0
   
    test_loader_iter = tqdm(test_loaders, total=len(test_loaders), desc="Test")
    for data1, data2, data3, target in test_loader_iter:
        Test_Data_name = test_loaders.dataset.sample_list[num].split('.h5')[0].split('/')[-1]
        if args.use_cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
            target = target.cuda()

        with torch.no_grad():
            data1 = tonemap(data1, args.input_tonemap)
            data2 = tonemap(data2, args.input_tonemap)
            data3 = tonemap(data3, args.input_tonemap)
            target = tonemap(target, args.label_tonemap)
            
            if args.offset: 
                # if output activation is tanh, then input should be normalized to [-1, 1]
                data1 = (data1 * 2.0) - 1.0
                data2 = (data2 * 2.0) - 1.0
                data3 = (data3 * 2.0) - 1.0
                target = (target * 2.0) - 1.0
            
            output = model(data1, data2, data3)
            
            if args.offset:
                output = (output + 1.0) / 2.0
                target = (target + 1.0) / 2.0
                
            output = tonemap(output, args.output_tonemap)
            
        # save the result to .H5 files
        hdrfile = h5py.File(outdir + "/" + Test_Data_name + '_hdr.h5', 'w')
        img = output[0, :, :, :]
        img = tv.utils.make_grid(img.data.cpu()).numpy()
        hdrfile.create_dataset('data', data=img)
        hdrfile.close()
        
        # save input/output as jpg files
        img = torch.squeeze(output*255.)
        img = img.data.cpu().numpy().astype(np.uint8)
        img = np.transpose(img, (2, 1, 0))
        img = img[:, :, [0, 1, 2]]
        imageio.imwrite(outdir + "/" + Test_Data_name + '_out.jpg', img)
        
        img = torch.squeeze(data1*255.)
        img = img.data.cpu().numpy().astype(np.uint8)
        img = np.transpose(img, (2, 1, 0))
        img = img[:, :, [0, 1, 2]]
        imageio.imwrite(outdir + "/" + Test_Data_name + '_data1.jpg', img)
        
        img = torch.squeeze(data2*255.)
        img = img.data.cpu().numpy().astype(np.uint8)
        img = np.transpose(img, (2, 1, 0))
        img = img[:, :, [0, 1, 2]]
        imageio.imwrite(outdir + "/" + Test_Data_name + '_data2.jpg', img)
        
        img = torch.squeeze(data3*255.)
        img = img.data.cpu().numpy().astype(np.uint8)
        img = np.transpose(img, (2, 1, 0))
        img = img[:, :, [0, 1, 2]]
        imageio.imwrite(outdir + "/" + Test_Data_name + '_data3.jpg', img)
        
        img = torch.squeeze(target*255.)
        img = img.data.cpu().numpy().astype(np.uint8)
        img = np.transpose(img, (2, 1, 0))
        img = img[:, :, [0, 1, 2]]
        imageio.imwrite(outdir + "/" + Test_Data_name + '_label.jpg', img)
        
        #########  Prepare to calculate metrics
        psnr_output = torch.squeeze(output[0].clone())
        psnr_target = torch.squeeze(target.clone())
        psnr_output = psnr_output.data.cpu().numpy().astype(np.float32)
        psnr_target = psnr_target.data.cpu().numpy().astype(np.float32)
        
        #########  Calculate metrics
        #psnr = psnr(psnr_output, psnr_target)
        psnr_norm = normalized_psnr(psnr_output, psnr_target, psnr_target.max())
        #psnr_mu = psnr_tanh_norm_mu_tonemap(psnr_target, psnr_output)

        val_psnr += psnr(psnr_output, psnr_target)
        val_psnr_norm += psnr_norm
        #val_psnr_mu += psnr_mu
        
        #hdr = torch.log(1 + 5000 * output.cpu()) / torch.log(
        #    Variable(torch.from_numpy(np.array([1 + 5000])).float()))
        #target = torch.log(1 + 5000 * target).cpu() / torch.log(
        #    Variable(torch.from_numpy(np.array([1 + 5000])).float()))

        #test_loss += F.mse_loss(hdr, target)
        test_loss += F.mse_loss(output, target)
            
        num = num + 1

    test_loss = test_loss / len(test_loaders.dataset)
    val_psnr = val_psnr / len(test_loaders.dataset)
    val_psnr = val_psnr_norm / len(test_loaders.dataset)
    #val_psnr_mu = val_psnr_mu / len(test_loaders.dataset)
    print('\n Test result - Average Loss: {:.4f}, Average PSNR: {:.4f}, Average PSNR_norm: {:.4f}'.format(test_loss.item(), val_psnr, val_psnr_norm))

    run_time = datetime.now().strftime('%m/%d %H:%M:%S')
    if not os.path.exists('./test_result.log'):
        with open('./test_result.log', 'w') as flog:
            flog.write('Model, Run_name, Epoch, Run_time, Test_loss, PSNR, PSNR_norm\n')
            flog.write(f'{args.model}, {args.run_name}, {args.epoch}, {run_time}, {test_loss:.6f}, {val_psnr:.6f}, {val_psnr_norm:.06f}\n')
    else:
        with open('./test_result.log', 'a') as flog:
            flog.write(f'{args.model}, {args.run_name}, {args.epoch}, {run_time}, {test_loss:.6f}, {val_psnr:.6f}, {val_psnr_norm:.06f}\n')
    
    return test_loss


def validation(epoch, model, valid_loaders, trained_model_dir, args):
    model.eval()
    val_psnr = 0
    val_psnr_mu = 0
    valid_loss = 0
    valid_num = len(valid_loaders)
    
    for batch_idx, (data1, data2, data3, target) in enumerate(valid_loaders):
        if args.use_cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
            target = target.cuda()
        
        with torch.no_grad():
            data1 = tonemap(data1, args.input_tonemap)
            data2 = tonemap(data2, args.input_tonemap)
            data3 = tonemap(data3, args.input_tonemap)
            target = tonemap(target, args.label_tonemap)
            
            if args.offset: 
                # if output activation is tanh, then input should be normalized to [-1, 1]
                data1 = (data1 * 2.0) - 1.0
                data2 = (data2 * 2.0) - 1.0
                data3 = (data3 * 2.0) - 1.0
                target = (target * 2.0) - 1.0
                
            output = model(data1, data2, data3)
            
            if args.offset:
                output = (output + 1.0) / 2.0
                target = (target + 1.0) / 2.0
            
            output = tonemap(output, args.output_tonemap)
    
        loss = F.l1_loss(output, target)
        valid_loss = valid_loss + loss
        
        #########  Prepare to calculate metrics
        psnr_output = torch.squeeze(output[0].clone())
        psnr_target = torch.squeeze(target.clone())
        psnr_output = psnr_output.data.cpu().numpy().astype(np.float32)
        psnr_target = psnr_target.data.cpu().numpy().astype(np.float32)
        
        #########  Calculate metrics
        psnr = normalized_psnr(psnr_output, psnr_target, psnr_target.max())
        #psnr_mu = psnr_tanh_norm_mu_tonemap(psnr_target, psnr_output)

        val_psnr = val_psnr + psnr
        #val_psnr_mu = val_psnr_mu + psnr_mu
        
    valid_loss = valid_loss / valid_num
    val_psnr = val_psnr / valid_num
    #val_psnr_mu = val_psnr_mu / valid_num
    print('Validation - Epoch {}: avg_loss: {:.4f}, Average PSNR: {:.4f}'.format(epoch, valid_loss, val_psnr))
    
    return valid_loss, val_psnr