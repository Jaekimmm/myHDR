import torch
import numpy as np
import time
import argparse
import torch.optim as optim
import torch.utils.data
from torch.nn import init
from dataset import DatasetFromHdf5

from model import *
from running_func import *
from utils import *
import os
from torchinfo import summary
from distutils.util import strtobool
import wandb

parser = argparse.ArgumentParser(description='Attention-guided HDR')

parser.add_argument('--model', type=str, default='LIGHTFUSE')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--train_data', default='./dataset_train')
parser.add_argument('--valid_data', default='./dataset_test')

parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')
parser.add_argument('--loss', type=str, default='L1')
parser.add_argument('--offset', action='store_true', default=False)
parser.add_argument('--label_tonemap', type=str, default=None)
parser.add_argument('--input_tonemap', type=str, default=None)
parser.add_argument('--output_tonemap', type=str, default=None)

parser.add_argument('--use_cuda', default=True)
parser.add_argument('--restore', default=True)
parser.add_argument('--load_model', default=True)

parser.add_argument('--seed', default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--epochs', type=int, default=800000)
parser.add_argument('--momentum', default=0.9)

args = parser.parse_args()

wandb = wandb.init(
    entity = "doobooteam",
    project = "myHDR",
    name = f"{args.model}-{args.run_name}",
    config = {
        "learning_rate": args.lr,
        "architecture": args.model,
        "dataset": args.train_data,
        "epochs" : args.epochs,
    }
)

torch.manual_seed(args.seed)

print("\n << CUDA devices >>")
if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    cuda_count = torch.cuda.device_count()
    print(f"Number of visible CUDA devices: {cuda_count}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
else:
    print("CUDA is not available.\n")
   
    
# Load data
train_loaders = torch.utils.data.DataLoader(
    data_loader(args.train_data, crop_size=256, geometry_aug=True),
    batch_size=args.batchsize, shuffle=True)
valid_loaders = torch.utils.data.DataLoader(
    data_loader(args.valid_data, crop_size=256, geometry_aug=False),
    batch_size=1, shuffle=False)


# make folders of trained model and result
if args.run_name:
    trained_model_dir = f"./trained-model-{args.model}-{args.run_name}/"
else:
    trained_model_dir = f"./trained-model-{args.model}/"
mk_dir(trained_model_dir)


# Get model
if args.model in globals(): 
    model = globals()[args.model]   # Find the model in the global namespace
model = model(args)
model.apply(weights_init_kaiming)   # Initialize the weights of the model
loss_model = VGGFeatureExtractor()  # Load VGG pre-trained model for loss calculation

if args.use_cuda: 
    model.cuda()
    loss_model.cuda()
    model = nn.DataParallel(model)
    loss_model = nn.DataParallel(loss_model)


print(f"[INFO] Start training with model {model}")
print(f"[INFO] Loss function : {args.loss}")
print(f"[INFO] Input preprocessing : Offset {args.offset}, Input tonemap {args.input_tonemap}, Label tonemap {args.label_tonemap}\n")


# Training loop
start_epoch = 0
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

## Check if there is a trained model to restore
if args.restore and len(os.listdir(trained_model_dir)):
    model, start_epoch = model_restore(model, trained_model_dir)
    print('[INFO] restart from epoch {}'.format(start_epoch))

start_train = time.time()
save_model_interval = max(1, args.epochs // 10)
for epoch in range(start_epoch + 1, args.epochs + 1):
    start = time.time()
    train_loss = train(epoch, model, loss_model, train_loaders, optimizer, trained_model_dir, wandb, args)
    end = time.time()
    #print(f'Epoch {epoch}/{args.epochs}, runtime : {(end-start):.4f} seconds, loss : {train_loss:.4f}')
    
    if epoch % save_model_interval == 0:
        model_name = trained_model_dir + 'trained_model{}.pkl'.format(epoch)
        torch.save(model.state_dict(), model_name)
        valid_loss, psnr = validation(epoch, model, valid_loaders, trained_model_dir, args)
       
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "psnr": psnr,
        })
        
end_train = time.time()

print(f"[INFO] Training finished. ({end_train - start_train:.4f} seconds)")
wandb.finish()
#save_plot(trained_model_dir)
