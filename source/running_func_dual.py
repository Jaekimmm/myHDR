import numpy as np
import os
import random
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
from datetime import datetime
import wandb
from tqdm import tqdm
import kornia
from sklearn.cluster import KMeans


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
    def __init__(self, data_dir, crop_size=0, crop_div=8, geometry_aug=False, map=''):
        super().__init__()
        self.crop_size = crop_size
        self.crop_div = crop_div
        self.geometry_aug = geometry_aug
        self.map_name = map
        
        self.data_dir = data_dir
        
        self.data_name = data_dir.split("_")
        if len(self.data_name) > 1:
            self.data_name = self.data_name[1]
        else:
            raise ValueError("Invalid dataset path format. Expected format: dataset_{name}_{test/train}.")
        
        self.sample_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.length = len(self.sample_list)
        
        print(f"[INFO] (data_loader) {self.length} data loaded from {data_dir}")
        if self.map_name != '':
            print(f"[INFO] (data_loader) {self.length} MAP_{self.map_name} loaded from {data_dir}")

    def __getitem__(self, index):
        sample_path = self.sample_list[index]
        sample_path = sample_path.strip()

        if os.path.exists(sample_path):
            if self.data_name == 'sice':
                with h5py.File(sample_path, 'r') as f:
                    data = np.stack((f['IN'][0:3, :, :],  # short
                                     f['IN'][0:3, :, :],  # mid
                                     f['IN'][3:6, :, :],  # long
                                     f['GT'][:])
                                    , axis=0)
            elif self.data_name == 'kalan':
                if self.map_name == '':
                    with h5py.File(sample_path, 'r') as f:
                        data = np.stack((f['IN'][3*3:4*3, :, :],  # short after gain adjustment
                                         f['IN'][4*3:5*3, :, :],  # mid after gain adjustment
                                         f['IN'][5*3:6*3, :, :],  # long after gain adjustment
                                         f['GT'][   :   , :, :])
                                        , axis=0)
                        exp = f['EXP'][:]
                else: 
                    with h5py.File(sample_path, 'r') as f:
                        data = np.stack((f['IN'][3*3:4*3, :, :],  # short after gain adjustment
                                         f['IN'][4*3:5*3, :, :],  # mid after gain adjustment
                                         f['IN'][5*3:6*3, :, :],  # long after gain adjustment
                                         f['GT'][   :   , :, :],
                                         f[f'MAP_{self.map_name}'][:, :, :])
                                        , axis=0)
                        exp = f['EXP'][:]
            
            if self.crop_div > 0 : 
                data = self.crop_for_div(data, self.crop_div)
            if self.crop_size > 0 : 
                data = self.imageCrop(data, self.crop_size)
            if self.geometry_aug : 
                data = self.image_Geometry_Aug(data)
            
            data_out = torch.from_numpy(data[0:4]).float()
            if self.map_name != '':
                map  = torch.from_numpy(data[4]).float()
            else:
                map = torch.zeros((1, 1, data.shape[2], data.shape[3]), dtype=torch.float32)
            
        
        return data_out, map, exp

    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)

    def imageCrop(self, data, crop_size):
        n, c, w, h = data.shape
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

        crop_data = data[:, :, start_w:end_w, start_h:end_h]
        return crop_data

    def image_Geometry_Aug(self, data):
        n, c, w, h = data.shape
        num = self.random_number(4)

        if num == 1:  # no aug
            in_data = data

        if num == 2:  # flip_left_right
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, :, index, :]

        if num == 3:  # flip_up_down
            index = np.arange(h, 0, -1) - 1
            in_data = data[:, :, :, index]

        if num == 4:  # rotate 180
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, :, index, :]
            index = np.arange(h, 0, -1) - 1
            in_data = in_data[:, :, :, index]

        return in_data
    
    def crop_for_div(self, data, crop_div=1):
        # crop the image to be divisible by crop_div
        num, channels, height, width = data.shape
        new_height = (height // crop_div) * crop_div
        new_width = (width // crop_div) * crop_div
        cropped_data = data[:, :, :new_height, :new_width]
    
        return cropped_data
        

def get_lr(epoch, lr, max_epochs):
    #if epoch <= max_epochs * 0.8:
    #    lr = lr
    #else:
    #    lr = 0.1 * lr
    return lr

def train(epoch, model, denoise_model, loss_func, gate_func, train_loaders, optimizer, trained_model_dir, wandb, args):
    # Adjust the learning rate (TODO : not used for now ...)
    #lr = get_lr(epoch, args.lr, args.epochs)
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = lr
    #print('[INFO] lr: {}'.format(optimizer.param_groups[0]['lr']))
    verbose = False
    
    log_file = os.path.join(trained_model_dir, 'train_args.log')
    with open(log_file, 'w') as flog:
        for key, value in vars(args).items():
            flog.write(f"{key}: {value}\n")
    
    model.train()
    denoise_model.eval()
    
    trainloss = 0
    avg_loss = 0
    loss_breakdown = {}
    avg_gated_ratio = 0
    skipped_batch = 0
    sample_num = random.randint(0, len(train_loaders)-1)
    
    train_loader_iter = tqdm(enumerate(train_loaders), total=len(train_loaders), desc=f"Epoch {epoch}")
    for batch_idx, (data, mask, exp) in train_loader_iter:
        if args.train_with_mask and mask.sum() == 0:
            if verbose: print(f"[LOG] batch{batch_idx} is skipped")
            skipped_batch += 1
            continue
        
        if verbose: print(f"[LOG] data to CUDA")
        if args.use_cuda:
            data = data.cuda()
            mask = mask.cuda()
        
        if args.adjust_exp_to is not None and args.adjust_exp_to != 'none':
            exp_time = {"short": 0, "mid": 1, "long": 2}
            data1 = data1 * (exp[exp_time[args.adjust_exp_to]]/exp[0])
            data2 = data2 * (exp[exp_time[args.adjust_exp_to]]/exp[1])
            data3 = data3 * (exp[exp_time[args.adjust_exp_to]]/exp[2])
        
        if verbose: print(f"[LOG] input tonemap")
        data1 = tonemap(data[:, 0], args.input_tonemap)
        data2 = tonemap(data[:, 1], args.input_tonemap)
        data3 = tonemap(data[:, 2], args.input_tonemap)
        target = tonemap(data[:, 3], args.label_tonemap)
        
        if (epoch == 1) and (batch_idx == sample_num):
            # print one sample data to check the image
            if verbose: print(f"[LOG] save sample images")
            save_tensor_to_img(data1[0, :, :, :], trained_model_dir + f'/sample_in1_b{batch_idx}.png')
            save_tensor_to_img(data2[0, :, :, :], trained_model_dir + f'/sample_in2_b{batch_idx}.png')
            save_tensor_to_img(data3[0, :, :, :], trained_model_dir + f'/sample_in3_b{batch_idx}.png')
            save_tensor_to_img(target[0, :, :, :], trained_model_dir + f'/sample_target_b{batch_idx}.png')
            #save_tensor_to_img(denoised_mid[0, :, :, :], trained_model_dir + f'/sample_denoised_mid_b{batch_idx}.png')
        
        if args.offset: 
            # if output activation is tanh, then input should be normalized to [-1, 1]
            data1 = (data1 * 2.0) - 1.0
            data2 = (data2 * 2.0) - 1.0
            data3 = (data3 * 2.0) - 1.0
            target = (target * 2.0) - 1.0
            
        if verbose: print(f"[LOG] model run and get output")
        optimizer.zero_grad()
        denoised_mid = denoise_model(data1, data2, data3)
        #output_raw = model(data1, data2, data3)
        output_raw = model(data1, denoised_mid, data3)
        
        if args.offset: 
            # if output activation is tanh, then input should be normalized to [-1, 1]
            output_raw = (output_raw + 1.0) / 2.0
            target = (target + 1.0) / 2.0
        
        if verbose: print(f"[LOG] output tonemap")
        output_raw = tonemap(output_raw, args.output_tonemap)
        
        # gating
        if verbose: print(f"[LOG] gating...")
        if args.load_gating_map == '':
            output, gated_ratio = gate_func(data1, denoised_mid, data3, output_raw, save_map=False, map_file=None)
        else:
            mask = torch.where(mask == args.map_index_enable, torch.ones_like(mask), torch.zeros_like(mask))
            mask = mask.expand_as(data1)
            
            if (epoch == 1) and (batch_idx == 0) and args.train_with_mask:
                save_tensor_to_img(mask[0, :, :, :], trained_model_dir + f'/sample_mask_b{batch_idx}.png')
            #output = output_raw * mask + data2 * (1 - mask)
            output = output_raw * mask + denoised_mid * (1 - mask)
            gated_ratio = (mask == args.map_index_enable).sum() / mask.numel()
        
        # Loss calculation
        if verbose: print(f"[LOG] loss calculation")
        if args.train_with_mask:
            output_masked = output * mask
            target_masked = target * mask
            loss, loss_dict = loss_func(output_masked, target_masked)
        else:
            #loss, loss_dict = loss_func(output_raw, target)
            loss, loss_dict = loss_func(output, target)
        
        if verbose: print(f"[LOG] back propagation")
        loss.backward()
        optimizer.step()
        trainloss = trainloss + loss
        avg_loss = avg_loss + loss
        avg_gated_ratio = avg_gated_ratio + gated_ratio
        
        if verbose: print(f"[LOG] get loss breakdown")
        for type, value in loss_dict.items():
            if type in loss_breakdown:
                loss_breakdown[type] += value
            else:
                loss_breakdown[type] = value
        
        train_loader_iter.set_postfix(loss=loss.item())
        
        if (batch_idx+1) % 10 == 0:
            trainloss = trainloss / 10
            loss_breakdown = {type: value / 10 for type, value in loss_breakdown.items()}
            wandb.log({
                "train_loss": loss.item(),
                "train_loss_breakdown": loss_breakdown
            })
            
            trainloss = 0
    
    avg_loss /= len(train_loaders)
    #print(f"skipped batch : {skipped_batch} / {len(train_loaders)}")
    
    #if (epoch == 1):
    #    print(f"[INFO] similarity : avg {np.mean(similiarity_hist):.4f}, min {np.min(similiarity_hist):.4f}, max {np.max(similiarity_hist):.4f}")
    
    return avg_loss

def testing_gate(model, denoise_model, loss_func, gate_func, test_loaders, outdir, args):
    model.eval()
    denoise_model.eval()
    verbose = False
    
    test_loss = 0
    val_psnr = 0
    val_psnr_norm = 0
    num = 0
    avg_gated_ratio = 0
    
    log_file = os.path.join(outdir, 'train_args.log')
    with open(log_file, 'w') as flog:
        for key, value in vars(args).items():
            flog.write(f"{key}: {value}\n")
    
    test_loader_iter = tqdm(test_loaders, total=len(test_loaders), desc="Test")
    for data, mask, exp in test_loader_iter:
    #for data, mask, exp, denoised_mid in test_loader_iter:
        Test_Data_name = test_loaders.dataset.sample_list[num].split('.h5')[0].split('/')[-1]
        if verbose: print(f"[LOG] data to CUDA")
        if args.use_cuda:
            data = data.cuda()
            mask = mask.cuda()
            #denoised_mid = denoised_mid.cuda()

        with torch.no_grad():
            if args.adjust_exp_to is not None and args.adjust_exp_to != 'none':
                exp_time = {"short": 0, "mid": 1, "long": 2}
                data1 = data1 * (exp[exp_time[args.adjust_exp_to]]/exp[0])
                data2 = data2 * (exp[exp_time[args.adjust_exp_to]]/exp[1])
                data3 = data3 * (exp[exp_time[args.adjust_exp_to]]/exp[2])
            
            if verbose: print(f"[LOG] tonemapping input data") 
            data1 = tonemap(data[:, 0], args.input_tonemap)
            data2 = tonemap(data[:, 1], args.input_tonemap)
            data3 = tonemap(data[:, 2], args.input_tonemap)
            target = tonemap(data[:,3], args.label_tonemap)
            
            if args.offset: 
                # if output activation is tanh, then input should be normalized to [-1, 1]
                data1 = (data1 * 2.0) - 1.0
                data2 = (data2 * 2.0) - 1.0
                data3 = (data3 * 2.0) - 1.0
                target = (target * 2.0) - 1.0
            
            if verbose: print(f"[LOG] model run")
            denoised_mid = denoise_model(data1, data2, data3)
            #output_raw = model(data1, data2, data3)
            output_raw = model(data1, denoised_mid, data3)
                
            if args.offset:
                output_raw = (output_raw + 1.0) / 2.0
                target = (target + 1.0) / 2.0
                
            if verbose: print(f"[LOG] tonmapping output data")
            output_raw = tonemap(output_raw, args.output_tonemap)
           
            # gating
            if verbose: print(f"[LOG] data masking")
            if args.load_gating_map == '':
                output, gated_ratio = gate_func(data1, denoised_mid, data3, output_raw, save_map=True, 
                                                map_file=f"{outdir}/{Test_Data_name}_gate_map_{args.output_gating}.png")
            else:
                mask = torch.where(mask == args.map_index_enable, torch.ones_like(mask), torch.zeros_like(mask))
                if 'ds' in args.map_preproc:
                    scale_factor = int(args.map_preproc.split('ds')[1])
                    mask = torch.nn.functional.interpolate(mask, scale_factor=1/scale_factor, mode='nearest')
                    mask = torch.nn.functional.interpolate(mask, scale_factor=scale_factor, mode='bilinear')
                elif 'box' in args.map_preproc:
                    kernel_size = int(args.map_preproc.split('box')[1])
                    mask = kornia.filters.box_blur(mask, (kernel_size, kernel_size))
                elif 'gaus' in args.map_preproc:
                    kernel_size = int(args.map_preproc.split('gaus')[1])
                    mask = kornia.filters.gaussian_blur2d(mask, (kernel_size, kernel_size), (1.5, 1.5))
                
                save_tensor_to_img(mask,       f"{outdir}/{Test_Data_name}_mask_{args.model}_{args.train_name}_{args.test_name}.png")
                mask = mask.expand_as(data1)
                #output = output_raw * mask + data2 * (1 - mask)
                output = output_raw * mask + denoised_mid * (1 - mask)
                gated_ratio = (mask == 2).sum() / mask.numel()
            
        # save the result to .H5 files
        if verbose: print(f"[LOG] Store the result")
        hdrfile = h5py.File(outdir + "/" + Test_Data_name + '_hdr.h5', 'w')
        img = output[0, :, :, :]
        img = tv.utils.make_grid(img.data.cpu()).numpy()
        hdrfile.create_dataset('data', data=img)
        hdrfile.close()
        
        # save input/output as jpg files
        
        if verbose: print(f"[LOG] Store images... out")
        save_tensor_to_img(output_raw, f"{outdir}/{Test_Data_name}_out_raw_{args.model}_{args.train_name}_{args.test_name}.png")
        if args.output_gating is not None and args.output_gating != 'none':
            save_tensor_to_img(output,     f"{outdir}/{Test_Data_name}_out_gated_{args.model}_{args.train_name}_{args.test_name}.png")
        
        if verbose: print(f"[LOG] Store images... input")
        #save_tensor_to_img(data1, f"{outdir}/{Test_Data_name}_data1.png")
        save_tensor_to_img(data2, f"{outdir}/{Test_Data_name}_data2.png")
        save_tensor_to_img(denoised_mid, f"{outdir}/{Test_Data_name}_denoised_mid.png")
        #save_tensor_to_img(data3, f"{outdir}/{Test_Data_name}_data3.png")
        save_tensor_to_img(target,f"{outdir}/{Test_Data_name}_label.png")
        
        #########  Prepare to calculate metrics
        if verbose: print(f"[LOG] Caculate metrics")
        psnr_output = torch.squeeze(output[:, :, 8:-8, 8:-8].clone())
        psnr_target = torch.squeeze(target[:, :, 8:-8, 8:-8].clone())
        psnr_output = psnr_output.data.cpu().numpy().astype(np.float32)
        psnr_target = psnr_target.data.cpu().numpy().astype(np.float32)
        
        #########  Calculate metrics
        psnr_norm = normalized_psnr(psnr_output, psnr_target, psnr_target.max())

        val_psnr += psnr(psnr_output, psnr_target)
        val_psnr_norm += psnr_norm
        
        loss, _ = loss_func(output, target)
        test_loss += loss
        num = num + 1
        
        avg_gated_ratio += gated_ratio
        if verbose: print(f"[LOG] Batch done...")

    test_loss = test_loss / len(test_loaders.dataset)
    val_psnr = val_psnr / len(test_loaders.dataset)
    val_psnr = val_psnr_norm / len(test_loaders.dataset)
    avg_gated_ratio = avg_gated_ratio / len(test_loaders.dataset)
    print(f"average: {avg_gated_ratio}")
    print('\n Test result - Average Loss: {:.4f}, Average PSNR: {:.4f}, Average PSNR_norm: {:.4f}, Average gated ratio: {:.4f}'.format(test_loss.item(), val_psnr, val_psnr_norm, avg_gated_ratio))

    run_time = datetime.now().strftime('%m/%d %H:%M:%S')
    if not os.path.exists('./test_result.log'):
        with open('./test_result.log', 'w') as flog:
            flog.write('Model, Train_name, Test_name, Epoch, Run_time, Test_loss, PSNR, PSNR_norm\n')
            flog.write(f'{args.model}, {args.train_name}, {args.test_name}, {args.epoch}, {run_time}, {test_loss:.6f}, {val_psnr:.6f}, {val_psnr_norm:.06f}\n')
    else:
        with open('./test_result.log', 'a') as flog:
            flog.write(f'{args.model}, {args.train_name}, {args.test_name}, {args.epoch}, {run_time}, {test_loss:.6f}, {val_psnr:.6f}, {val_psnr_norm:.06f}, {avg_gated_ratio:.04f}\n')
    
    return test_loss


def validation(epoch, model, gate_func, valid_loaders, trained_model_dir, args):
    model.eval()
    
    val_psnr = 0
    valid_loss = 0
    valid_num = len(valid_loaders)
    
    valid_loader_iter = tqdm(valid_loaders, total=len(valid_loaders), desc="Test")
    for data, mask, exp in valid_loader_iter:
        if args.use_cuda:
            data = data.cuda()
            mask = mask.cuda()
        
        with torch.no_grad():
            if args.adjust_exp_to is not None and args.adjust_exp_to != 'none':
                exp_time = {"short": 0, "mid": 1, "long": 2}
                data1 = data1 * (exp[exp_time[args.adjust_exp_to]]/exp[0])
                data2 = data2 * (exp[exp_time[args.adjust_exp_to]]/exp[1])
                data3 = data3 * (exp[exp_time[args.adjust_exp_to]]/exp[2])
            
            data1 = tonemap(data[:, 0], args.input_tonemap)
            data2 = tonemap(data[:, 1], args.input_tonemap)
            data3 = tonemap(data[:, 2], args.input_tonemap)
            target = tonemap(data[:,3], args.label_tonemap)
            
            if args.offset: 
                # if output activation is tanh, then input should be normalized to [-1, 1]
                data1 = (data1 * 2.0) - 1.0
                data2 = (data2 * 2.0) - 1.0
                data3 = (data3 * 2.0) - 1.0
                target = (target * 2.0) - 1.0
                
            output_raw = model(data1, data2, data3)
            
            if args.offset:
                output_raw = (output_raw + 1.0) / 2.0
                target = (target + 1.0) / 2.0
            
            output_raw = tonemap(output_raw, args.output_tonemap)
            
            # gating
            if args.load_gating_map == '':
                output, _ = gate_func(data1, data2, data3, output_raw, save_map=False, map_file=None)
            else:
                mask = torch.where(mask == args.map_index_enable, torch.ones_like(mask), torch.zeros_like(mask))
                mask = mask.expand_as(data1)
                output = output_raw * mask + data2 * (1 - mask)
            
    
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


## Loss related
class get_loss_function(nn.Module):
    def __init__(self, args):
        super(get_loss_function, self).__init__()
        
        self.loss_types = args.loss if isinstance(args.loss, list) else [args.loss]
        self.weights = args.loss_weights if isinstance(args.loss_weights, list) else [args.loss_weights]
        
        if len(self.loss_types) > 1 and len(self.loss_types) != len(self.weights):
            raise ValueError(f"[WARN] # of loss type ({len(self.loss_types)}) != # of weight ({len(self.weights)})")
        print(f"[INFO] Loss function : {list(zip(self.loss_types, self.weights))}")
        
        self.loss_func = [loss_type(l.lower(), args) for l in self.loss_types]
        
    def forward(self, input, target):
        total_loss = 0.0
        loss_dict = {loss_type: 0.0 for loss_type in self.loss_types}
 
        for loss_type, loss_func, weight in zip (self.loss_types, self.loss_func, self.weights):
            loss_value = loss_func(input, target)
            loss_dict[loss_type] += weight * loss_value.item()
            total_loss += weight * loss_value
        return total_loss, loss_dict

#region : loss sub-function
def loss_type(loss_type, args):
    if loss_type == 'mse':
        print(f"[INFO] Get loss type : MSE")
        return nn.MSELoss()
    elif loss_type == 'bce':
        print(f"[INFO] Get loss type : BCE")
        return nn.BCELoss()
    elif loss_type == 'dice':
        print(f"[INFO] Get loss type : DICE")
        return DiceLoss()
    elif loss_type.lower() == 'l1':
        print(f"[INFO] Get loss type : L1")
        return nn.L1Loss()
    elif loss_type == 'vgg':
        loss_model = VGGFeatureExtractor(args.vgg_layer)
        loss_model.cuda()
        loss_model = nn.DataParallel(loss_model)
        loss_model.eval()
        return VGGLoss(loss_model)
    elif loss_type == 'lab':
        return LABLoss('lab', 'unsignedL', 'L1')
    elif loss_type == 'ab':
        return LABLoss('ab', 'unsignedL', 'L1')
    elif loss_type == 'lab_lonly_signed':
        return LABLoss('l', 'unsignedL', 'L1')
    elif loss_type == 'lab_lonly_unsigned':
        return LABLoss('l', 'unsignedL', 'L1')
    elif loss_type == 'vgg_mult_l1':
        loss_model = VGGFeatureExtractor(args.vgg_layer)
        loss_model.cuda()
        loss_model = nn.DataParallel(loss_model)
        loss_model.eval()
        return VGGmultL1_Loss(loss_model)
    elif loss_type == 'ssim':
        return kornia.losses.SSIMLoss(window_size=11)
    else:
        raise ValueError(f"[ERROR] Invalid loss type ({loss_type})")
        
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1e-6
        print(f"[INFO] Get loss type : DICE")
    def forward(self, input, target):
        intersection = (input * target).sum()
        dice = 1 - (2. * intersection + self.eps) / (input.sum() + target.sum() + self.eps)
        return 1 - dice
          
class VGGLoss(nn.Module):
    def __init__(self, vgg_model):
        super(VGGLoss, self).__init__()
        self.vgg_model = vgg_model
        print(f"[INFO] Get loss type : VGG")
        print(f"[INFO] VGG model for loss : {self.vgg_model}")
    
    def forward(self, input, target):
        input_features = self.vgg_model(input)
        target_features = self.vgg_model(target)
        return F.l1_loss(input_features, target_features)
    
class VGGmultL1_Loss(nn.Module):
    def __init__(self, vgg_model):
        super(VGGmultL1_Loss, self).__init__()
        self.vgg_model = vgg_model
        self.l1_loss = nn.L1Loss()
        print(f"[INFO] Get loss type : VGGmultL1")
        print(f"[INFO] VGG model for loss : {self.vgg_model}")
        
    def forward(self, input, target):
        input_features = self.vgg_model(input)
        target_features = self.vgg_model(target)
        vgg_loss = self.l1_loss(input_features, target_features)
        lab_loss = self.l1_loss(input, target)
        return vgg_loss * lab_loss * 10 

class LABLoss(nn.Module):
    def __init__(self, channel, norm, loss_type):
        super(LABLoss, self).__init__()
        self.channel = channel.lower()
        self.norm = norm.lower()
        if loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif loss_type == 'L1':
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()
        print(f"[INFO] Get loss type : LAB")
        print(f"       (ch - {self.channel}, norm - {self.norm}, type - {self.loss}")
        
    def forward(self, input, target):
        if self.norm == 'unsignedl':
            input_lab  = rgb_to_lab_norm2(input)
            target_lab = rgb_to_lab_norm2(target)
        elif self.norm == 'signedl':
            input_lab  = rgb_to_lab_norm(input)
            target_lab = rgb_to_lab_norm(target)
        else:
            input_lab  = kornia.color.RgbToLab()(input)
            target_lab = kornia.color.RgbToLab()(target)
    
        if self.channel == 'l':
            input_  =  input_lab[:, 0:1, :, :]
            target_ = target_lab[:, 0:1, :, :]
        elif self.channel == 'ab':
            input_  =  input_lab[:, 1:3, :, :]
            target_ = target_lab[:, 1:3, :, :]
        else:
            input_  =  input_lab[:, 0:3, :, :]
            target_ = target_lab[:, 0:3, :, :]
            
        return self.loss(input_, target_)
#endregion


class get_gating_scheme:
    def __init__(self, type):
        self.type = type.lower()
        
        if self.type == 'none' or self.type == 'load_map':
            self.gate_func = self._no_gating
        elif self.type == 'l1_thr_bin':
            self.gate_func = self._l1_thr_bin
        elif self.type == 'l1_thr_blend':
            self.gate_func = self._l1_thr_blend
        elif self.type == 'l1_blend':
            self.gate_func = self._l1_blend
        elif self.type == 'l1_sat_short':
            self.gate_func = self._l1_sat_short
        elif self.type == 'l1_sat_mid':
            self.gate_func = self._l1_sat_mid
        elif self.type == 'ssim_thr':
            self.gate_func = self._ssim_threshold
        elif self.type == 'ssim_thr_blend':
            self.gate_func = self._ssim_thr_blend
        elif self.type == 'ssim_blend':
            self.gate_func = self._ssim_blend
        elif self.type == 'gt_map':
            self.gate_func = self._gt_map
        else:
            print(f"[WARNING] Invalid gating scheme ({self.type}) set to 'none'")
            self.gate_func = self._no_gating
        
        print(f"[INFO] Gating scheme : {self.type}")
        
    def __call__(self, short, mid, long, output, save_map, map_file) -> torch.Tensor:
        return self.gate_func(short, mid, long, output, save_map, map_file)
    
    def _no_gating(self, short, mid, long, output, save_map=False, map_file=None):
        return output, 0
    
    def _l1_thr_bin(self, short, mid, long, output, save_map=False, map_file=None):
        thr = 0.5
        l1_short = torch.abs(mid - short).sum(dim=1, keepdim=True)
        l1_long  = torch.abs(mid - long).sum(dim=1, keepdim=True)
        gate_map = torch.where(l1_short > l1_long, l1_short, l1_long)
        gate_map = torch.where(gate_map < thr, torch.zeros_like(gate_map), torch.ones_like(gate_map))
        gated_ratio = (gate_map == 0).sum() / gate_map.numel()
        gate_map = gate_map.expand_as(mid)
        
        if save_map:
            save_tensor_to_img(gate_map, map_file)
        
        return (gate_map * output + (1 - gate_map) * mid), gated_ratio
    
    def _l1_thr_blend(self, short, mid, long, output, save_map=False, map_file=None):
        thr = 0.5
        l1_short = torch.abs(mid - short).sum(dim=1, keepdim=True)
        l1_long  = torch.abs(mid -  long).sum(dim=1, keepdim=True)
        gate_map = torch.where(l1_short > l1_long, l1_short, l1_long).clamp(min=0, max=1) # norm with max?
        gate_map = torch.where(gate_map < thr, torch.zeros_like(gate_map), gate_map)
        gated_ratio = (gate_map == 0).sum() / gate_map.numel()
        gate_map = gate_map.expand_as(mid)
        
        if save_map:
            save_tensor_to_img(gate_map, map_file)
        
        return (gate_map * output + (1 - gate_map) * mid), gated_ratio
    
    def _l1_blend(self, short, mid, long, output, save_map=False, map_file=None):
        l1_short = torch.abs(mid - short).mean(dim=1, keepdim=True)
        l1_long  = torch.abs(mid - long).mean(dim=1, keepdim=True)
        gate_map = (l1_short + l1_long) / 2
        gated_ratio = (gate_map == 0).sum() / gate_map.numel()
        gate_map = gate_map.expand_as(mid)
        
        if save_map:
            save_tensor_to_img(gate_map, map_file)
        
        return (gate_map * output + (1 - gate_map) * mid), gated_ratio
   
    def _l1_sat_short_bin_gaussian(self, short, mid, long, output, save_map=False, map_file=None):
        thr = 0.5
        short_mid_exp = (mid * 2).clamp(min=0, max=1) / 2
        l1_short = torch.abs(mid - short).sum(dim=1, keepdim=True)
        short_saturated = torch.abs(short - short_mid_exp).mean(dim=1, keepdim=True)
        gate_map = short_saturated + l1_short
        #gate_map = gate_map.clamp(min=0, max=1)
        gate_map = torch.where(gate_map < thr, torch.zeros_like(gate_map), torch.ones_like(gate_map))
        gauss = gaussian_2d(window_size=3, channel=3, sigma=1.5, device=short.device)
        gate_map = F.conv2d(gate_map, gauss, padding=1, groups=3)
        gated_ratio = (gate_map == 0).sum() / gate_map.numel()
        print(gated_ratio)
        gate_map = gate_map.expand_as(mid)
        
        if save_map:
            save_tensor_to_img(gate_map, map_file)
        
        return (gate_map * output + (1 - gate_map) * mid), gated_ratio
    
    def _l1_sat_short(self, short, mid, long, output, save_map=False, map_file=None):
        thr = 0.5
        short_mid_exp = (short * 2).clamp(min=0, max=1) / 2
        l1_short = torch.abs(mid - short).sum(dim=1, keepdim=True)
        short_saturated = torch.abs(short - short_mid_exp).mean(dim=1, keepdim=True)
        gate_map = short_saturated + l1_short
        gate_map = gate_map.clamp(min=0, max=1)
        gate_map = torch.where(gate_map < thr, torch.zeros_like(gate_map), gate_map)
        gated_ratio = (gate_map == 0).sum() / gate_map.numel()
        print(gated_ratio)
        gate_map = gate_map.expand_as(mid)
        
        if save_map:
            save_tensor_to_img(gate_map, map_file)
        
        return (gate_map * output + (1 - gate_map) * mid), gated_ratio
    
    def _l1_sat_mid(self, short, mid, long, output, save_map=False, map_file=None):
        thr = 0.5
        mid_with_long_exp = (mid * 2).clamp(min=0, max=1) / 2
        l1_short = torch.abs(mid - short).sum(dim=1, keepdim=True)
        short_saturated = torch.abs(short - mid_with_long_exp).mean(dim=1, keepdim=True)
        gate_map = short_saturated + l1_short
        gate_map = gate_map.clamp(min=0, max=1)
        gate_map = torch.where(gate_map < thr, torch.zeros_like(gate_map), gate_map)
        gated_ratio = (gate_map == 0).sum() / gate_map.numel()
        print(gated_ratio)
        gate_map = gate_map.expand_as(mid)
        
        if save_map:
            save_tensor_to_img(gate_map, map_file)
        
        return (gate_map * output + (1 - gate_map) * mid), gated_ratio
    
    def _ssim_threshold(self, short, mid, long, output, save_map=False, map_file=None):
        thr = 0.5
        ssim_short, _ = ssim(mid, short, window_size=11)
        ssim_long,  _ = ssim(mid, long, window_size=11)
        gate_map = (ssim_short + ssim_long) / 2
        gate_map = (gate_map >= thr).float()
        gate_map = gate_map.expand_as(mid)
        gated_ratio = (gate_map == 0).sum() / gate_map.numel()
        print(f"[INFO] Gated ratio : {gated_ratio:.4f}")
        
        if save_map:
            save_tensor_to_img(gate_map, map_file)
        
        return (1 - gate_map) * output + gate_map * mid 
    
    def _ssim_thr_blend(self, short, mid, long, output, save_map=False, map_file=None):
        thr = 0.7
        ssim_short, _ = ssim(mid, short, window_size=11)
        ssim_long,  _ = ssim(mid, long, window_size=11)
        gate_map = (ssim_short + ssim_long) / 2
        gate_map = torch.where(gate_map >= thr, torch.ones_like(gate_map), gate_map)
        gate_map = gate_map.expand_as(mid)
        gated_ratio = (gate_map == 1).sum() / gate_map.numel()
        print(f"[INFO] Gated ratio : {gated_ratio:.4f}")
        
        if save_map:
            save_tensor_to_img(gate_map, map_file)
        
        return (1 - gate_map) * output + gate_map * mid 
    
    def _ssim_blend(self, short, mid, long, output, save_map=False, map_file=None):
        ssim_short, _ = ssim(mid, short, window_size=11)
        ssim_long,  _ = ssim(mid, long, window_size=11)
        gate_map = (ssim_short + ssim_long) / 2
        gate_map = gate_map.expand_as(mid)
        gated_ratio = (gate_map == 0).sum() / gate_map.numel()
        print(f"[INFO] Gated ratio : {gated_ratio:.4f}")
        
        if save_map:
            save_tensor_to_img(gate_map, map_file)
        
        return (1 - gate_map) * output + gate_map * mid 
    
    def _gt_map(self, short, mid, gate_map, output, save_map=False, map_file=None):
        gated_ratio = (gate_map == 0).sum() / gate_map.numel()
        print(f"[INFO] Gated ratio : {gated_ratio:.4f}")
        
        if save_map:
            save_tensor_to_img(gate_map, map_file)
        
        return (gate_map * output + (1 - gate_map) * mid), gated_ratio
        
def get_map_by_hist_kmean(diff_tensor):
    diff = diff_tensor.squeeze().cpu().numpy()  # shape: (H, W)
    flat_diff = diff.reshape(-1, 1)  # shape: (H*W, 1)

    # K-means clustering (2 clusters: low-change, high-change)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(flat_diff)  # shape: (H*W,)

    # 어떤 클러스터가 'HDR 변화 큰 쪽'인지 확인
    centers = kmeans.cluster_centers_.squeeze()
    high_change_label = np.argmax(centers)

    # 이진 마스크 생성
    binary_mask = (labels == high_change_label).astype(np.uint8).reshape(diff_tensor.shape)
    
    return binary_mask    

def train_map(epoch, model, loss_func, gate_func, train_loaders, optimizer, trained_model_dir, wandb, args):
    # Adjust the learning rate (TODO : not used for now ...)
    #lr = get_lr(epoch, args.lr, args.epochs)
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = lr
    #print('[INFO] lr: {}'.format(optimizer.param_groups[0]['lr']))
    verbose = False
    
    log_file = os.path.join(trained_model_dir, 'train_args.log')
    with open(log_file, 'w') as flog:
        for key, value in vars(args).items():
            flog.write(f"{key}: {value}\n")
    
    model.train()
    
    trainloss = 0
    avg_loss = 0
    loss_breakdown = {}
    avg_gated_ratio = 0
    skipped_batch = 0
    
    train_loader_iter = tqdm(enumerate(train_loaders), total=len(train_loaders), desc=f"Epoch {epoch}")
    for batch_idx, (data, mask) in train_loader_iter:
        #if args.train_with_mask and mask.sum() == 0:
        #    if verbose: print(f"[LOG] batch{batch_idx} is skipped")
        #    skipped_batch += 1
        #    continue
        
        if verbose: print(f"[LOG] data to CUDA")
        if args.use_cuda:
            data = data.cuda()
            mask = mask.cuda()
        
        
        if verbose: print(f"[LOG] input tonemap")
        data1 = tonemap(data[:, 0], args.input_tonemap)
        data2 = tonemap(data[:, 1], args.input_tonemap)
        data3 = tonemap(data[:, 2], args.input_tonemap)
        #target = tonemap(data[:, 3], args.label_tonemap)
        
        if (epoch == 1) and (batch_idx == 0):
            # print one sample data to check the image
            if verbose: print(f"[LOG] save sample images")
            save_tensor_to_img(data1[0, :, :, :], trained_model_dir + f'/sample_in1_b{batch_idx}.png')
            save_tensor_to_img(data2[0, :, :, :], trained_model_dir + f'/sample_in2_b{batch_idx}.png')
            save_tensor_to_img(data3[0, :, :, :], trained_model_dir + f'/sample_in3_b{batch_idx}.png')
            #save_tensor_to_img(target[0, :, :, :], trained_model_dir + f'/sample_target_b{batch_idx}.png')
        
        if args.offset: 
            # if output activation is tanh, then input should be normalized to [-1, 1]
            data1 = (data1 * 2.0) - 1.0
            data2 = (data2 * 2.0) - 1.0
            data3 = (data3 * 2.0) - 1.0
            #target = (target * 2.0) - 1.0
            
        if verbose: print(f"[LOG] model run and get output")
        optimizer.zero_grad()
        output_raw = model(data1, data2, data3)
        
        #if args.offset: 
        #    # if output activation is tanh, then input should be normalized to [-1, 1]
        #    output_raw = (output_raw + 1.0) / 2.0
        #    #target = (target + 1.0) / 2.0
        
        #if verbose: print(f"[LOG] output tonemap")
        #output_raw = tonemap(output_raw, args.output_tonemap)
        
        ## gating
        #if verbose: print(f"[LOG] gating...")
        #if args.load_gating_map == '':
        #    output, gated_ratio = gate_func(data1, data2, data3, output_raw, save_map=False, map_file=None)
        #else:
        #    mask = torch.where(mask == args.map_index_enable, torch.ones_like(mask), torch.zeros_like(mask))
        #    mask = mask.expand_as(data1)
        #    output = output_raw * mask + data2 * (1 - mask)
        #    gated_ratio = (mask == args.map_index_enable).sum() / mask.numel()
        #    if (epoch == 1) and (batch_idx == 0):
        #        if args.load_gating_map is not '':
        #            save_tensor_to_img(mask[0, : :, :], trained_model_dir + f'/sample_mask_b{batch_idx}.png')
        
        # Loss calculation
        #if verbose: print(f"[LOG] loss calculation")
        #if args.train_with_mask:
        #    output_masked = output_raw * mask
        #    target_masked = target * mask
        #    loss, loss_dict = loss_func(output_masked, target_masked)
        #else:
        #    loss, loss_dict = loss_func(output_raw, target)
        loss, loss_dict = loss_func(output_raw, mask)
        
        if verbose: print(f"[LOG] back propagation")
        loss.backward()
        optimizer.step()
        trainloss = trainloss + loss
        avg_loss = avg_loss + loss
        #avg_gated_ratio = avg_gated_ratio + gated_ratio
        
        if verbose: print(f"[LOG] get loss breakdown")
        for type, value in loss_dict.items():
            if type in loss_breakdown:
                loss_breakdown[type] += value
            else:
                loss_breakdown[type] = value
        
        train_loader_iter.set_postfix(loss=loss.item())
        
        if (batch_idx+1) % 1 == 0:
            trainloss = trainloss / 1
            loss_breakdown = {type: value / 1 for type, value in loss_breakdown.items()}
            wandb.log({
                "train_loss": trainloss.item(),
                "train_loss_breakdown": loss_breakdown
            })
            
            trainloss = 0
    
    avg_loss /= len(train_loaders)
    #print(f"skipped batch : {skipped_batch} / {len(train_loaders)}")
    
    #if (epoch == 1):
    #    print(f"[INFO] similarity : avg {np.mean(similiarity_hist):.4f}, min {np.min(similiarity_hist):.4f}, max {np.max(similiarity_hist):.4f}")
    
    return avg_loss

def testing_map(model, loss_func, gate_func, test_loaders, outdir, args):
    model.eval()
    verbose = False
    
    test_loss = 0
    val_psnr = 0
    val_psnr_norm = 0
    num = 0
    avg_gated_ratio = 0
    
    test_loader_iter = tqdm(test_loaders, total=len(test_loaders), desc="Test")
    for data, mask in test_loader_iter:
        Test_Data_name = test_loaders.dataset.sample_list[num].split('.h5')[0].split('/')[-1]
        if verbose: print(f"[LOG] data to CUDA")
        if args.use_cuda:
            data = data.cuda()
            mask = mask.cuda()

        with torch.no_grad():
            
            if verbose: print(f"[LOG] tonemapping input data") 
            data1 = tonemap(data[:, 0], args.input_tonemap)
            data2 = tonemap(data[:, 1], args.input_tonemap)
            data3 = tonemap(data[:, 2], args.input_tonemap)
            #target = tonemap(data[:,3], args.label_tonemap)
            
            if args.offset: 
                # if output activation is tanh, then input should be normalized to [-1, 1]
                data1 = (data1 * 2.0) - 1.0
                data2 = (data2 * 2.0) - 1.0
                data3 = (data3 * 2.0) - 1.0
                #target = (target * 2.0) - 1.0
            
            if verbose: print(f"[LOG] model run")
            output_raw = model(data1, data2, data3)
            output = torch.where(output_raw > 0.3, torch.ones_like(output_raw), torch.zeros_like(output_raw))
                
            #if args.offset:
            #    output_raw = (output_raw + 1.0) / 2.0
            #    target = (target + 1.0) / 2.0
                
            #if verbose: print(f"[LOG] tonmapping output data")
            #output_raw = tonemap(output_raw, args.output_tonemap)
           
            ## gating
            #if verbose: print(f"[LOG] data masking")
            #if args.load_gating_map == '':
            #    output, gated_ratio = gate_func(data1, data2, data3, output_raw, save_map=True, 
            #                                    map_file=f"{outdir}/{Test_Data_name}_gate_map_{args.gating}.png")
            #else:
            #    mask = torch.where(mask == args.map_index_enable, torch.ones_like(mask), torch.zeros_like(mask))
            #    save_tensor_to_img(mask,       f"{outdir}/{Test_Data_name}_mask.png")
            #    mask = mask.expand_as(data1)
            #    output = output_raw * mask + data2 * (1 - mask)
            #    gated_ratio = (mask == 2).sum() / mask.numel()
            
        # save the result to .H5 files
        #if verbose: print(f"[LOG] Store the result")
        #hdrfile = h5py.File(outdir + "/" + Test_Data_name + '_hdr.h5', 'w')
        #img = output[0, :, :, :]
        #img = tv.utils.make_grid(img.data.cpu()).numpy()
        #hdrfile.create_dataset('data', data=img)
        #hdrfile.close()
        
        # save input/output as jpg files
        #save_jpg(output_raw, f"{outdir}/{Test_Data_name}_out_raw.jpg")
        #save_jpg(output,     f"{outdir}/{Test_Data_name}_out_gated_{args.gating}.jpg")
        #   
        #save_jpg(data1, f"{outdir}/{Test_Data_name}_data1.jpg")
        #save_jpg(data2, f"{outdir}/{Test_Data_name}_data2.jpg")
        #save_jpg(data3, f"{outdir}/{Test_Data_name}_data3.jpg")
        #save_jpg(target,f"{outdir}/{Test_Data_name}_label.jpg")
        
        if verbose: print(f"[LOG] Store images... out")
        #save_tensor_to_img(output_raw, f"{outdir}/{Test_Data_name}_out_prob.png")
        save_tensor_to_img(output,     f"{outdir}/{Test_Data_name}_out_mask_0p3.png")
        
        if verbose: print(f"[LOG] Store images... input")
        #save_tensor_to_img(data1, f"{outdir}/{Test_Data_name}_data1.png")
        #save_tensor_to_img(data2, f"{outdir}/{Test_Data_name}_data2.png")
        #save_tensor_to_img(data3, f"{outdir}/{Test_Data_name}_data3.png")
        #save_tensor_to_img(mask,  f"{outdir}/{Test_Data_name}_label.png")
        
        #########  Prepare to calculate metrics
        #if verbose: print(f"[LOG] Caculate metrics")
        #psnr_output = torch.squeeze(output[0].clone())
        #psnr_target = torch.squeeze(target.clone())
        #psnr_output = psnr_output.data.cpu().numpy().astype(np.float32)
        #psnr_target = psnr_target.data.cpu().numpy().astype(np.float32)
        
        ##########  Calculate metrics
        #psnr_norm = normalized_psnr(psnr_output, psnr_target, psnr_target.max())

        #val_psnr += psnr(psnr_output, psnr_target)
        #val_psnr_norm += psnr_norm
        
        loss, _ = loss_func(output_raw, mask)
        test_loss += loss
        num = num + 1
        
        #avg_gated_ratio += gated_ratio
        if verbose: print(f"[LOG] Batch done...")

    test_loss = test_loss / len(test_loaders.dataset)
    #val_psnr = val_psnr / len(test_loaders.dataset)
    #val_psnr = val_psnr_norm / len(test_loaders.dataset)
    #avg_gated_ratio = avg_gated_ratio / len(test_loaders.dataset)
    #print(f"average: {avg_gated_ratio}")
    #print('\n Test result - Average Loss: {:.4f}, Average PSNR: {:.4f}, Average PSNR_norm: {:.4f}, Average gated ratio: {:.4f}'.format(test_loss.item(), val_psnr, val_psnr_norm, avg_gated_ratio))

    run_time = datetime.now().strftime('%m/%d %H:%M:%S')
    if not os.path.exists('./test_map_result.log'):
        with open('./test_map_result.log', 'w') as flog:
            flog.write('Model, Train_name, Test_name, Epoch, Run_time, Test_loss\n')
            flog.write(f'{args.model}, {args.train_name}, {args.test_name}, {args.epoch}, {run_time}, {test_loss:.6f}\n')
    else:
        with open('./test_map_result.log', 'a') as flog:
            flog.write(f'{args.model}, {args.train_name}, {args.test_name}, {args.epoch}, {run_time}, {test_loss:.6f}\n')
    
    return test_loss
