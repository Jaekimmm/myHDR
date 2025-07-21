import numpy as np
import torch
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
from distutils.util import strtobool
import wandb
import torch.onnx
from torch.export import export
import shutil

parser = argparse.ArgumentParser(description='Attention-guided HDR')

parser.add_argument('--train_name', type=str, default='')
parser.add_argument('--train_data', default='./dataset_train')
parser.add_argument('--valid_data', default='./dataset_test')

parser.add_argument('--model', type=str, default='LIGHTFUSE')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')

parser.add_argument('--loss', nargs='+', type=str, default=['MSE', 'VGG'])
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1, 1])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--init', type=str, default='kaiming')
parser.add_argument('--batchsize', type=int, default=32)

parser.add_argument('--train_with_mask', action='store_true', default=False)
parser.add_argument('--load_gating_map', type=str, default='')
parser.add_argument('--map_index_enable', type=int, default=1)
parser.add_argument('--output_gating', type=str, default='none')


parser.add_argument('--offset', action='store_true', default=False)
parser.add_argument('--input_tonemap', type=str, default=None)
parser.add_argument('--label_tonemap', type=str, default=None)
parser.add_argument('--output_tonemap', type=str, default=None)
parser.add_argument('--adjust_exp_to', type=str, default=None)

parser.add_argument('--use_cuda', default=True)
parser.add_argument('--restore', default=True)
parser.add_argument('--load_model', default=True)

parser.add_argument('--seed', default=1)
parser.add_argument('--vgg_layer', type=int, default=22)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', default=0.9)

args = parser.parse_args()

wandb = wandb.init(
    entity = "doobooteam",
    project = "myHDR-final",
    name = f"{args.model}-{args.train_name}",
    config = {
        "learning_rate": args.lr,
        "architecture": args.model,
        "dataset": args.train_data,
        "epochs" : args.epochs,
        "batchsize" : args.batchsize,
    },
    #mode='disabled'
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
    data_loader(args.train_data, geometry_aug=True, map=args.load_gating_map),
    batch_size=args.batchsize, shuffle=True, num_workers=cuda_count)
valid_loaders = torch.utils.data.DataLoader(
    data_loader(args.valid_data, crop_size=256, geometry_aug=False, map=args.load_gating_map),
    batch_size=1, shuffle=False, num_workers=cuda_count)


# make folders of trained model and result
if args.train_name:
    trained_model_dir = f"./trained-model-{args.model}-{args.train_name}/"
else:
    trained_model_dir = f"./trained-model-{args.model}/"
mk_dir(trained_model_dir)


# Get model
if args.model in globals(): 
    model = globals()[args.model]   # Find the model in the global namespace
model = model(args)
if args.init == 'zero':
    model.apply(weights_init_zero)
elif args.init == 'average3x3':
    model.apply(weights_init_average3x3)
else:
    model.apply(weights_init_kaiming)   # Initialize the weights of the model

print(f"[INFO] Start training with model {model}")
print(f"[INFO] Input preprocessing : Offset {args.offset}, Input tonemap {args.input_tonemap}, Label tonemap {args.label_tonemap}")

if args.use_cuda: 
    model.cuda()
    model = nn.DataParallel(model)


# Training loop
start_epoch = 0
best_psnr = float('-inf')
best_epoch = 0
best_model_path = ''

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
loss_func = get_loss_function(args)

if args.load_gating_map == '':
    gate_func = get_gating_scheme('load_map')
else:
    gate_func = get_gating_scheme(args.output_gating)

## Check if there is a trained model to restore
if args.restore and len(os.listdir(trained_model_dir)):
    model, start_epoch = model_restore(model, trained_model_dir)
    print('[INFO] restart from epoch {}'.format(start_epoch))

start_train = time.time()
save_model_interval = max(1, args.epochs // 20)
for epoch in range(start_epoch + 1, args.epochs + 1):
    start = time.time()
    train_loss = train(epoch, model, loss_func, gate_func, train_loaders, optimizer, trained_model_dir, wandb, args)
    
    if epoch % save_model_interval == 0:
        model_name = trained_model_dir + 'trained_model{}.pkl'.format(epoch)
        torch.save(model.state_dict(), model_name)
        valid_loss, psnr = validation(epoch, model, gate_func, valid_loaders, trained_model_dir, args)
       
        wandb.log({
            "valid_loss": valid_loss,
            "psnr": psnr,
        })
        
        if psnr > best_psnr:
            best_psnr = psnr
            best_epoch = epoch
            best_model_path = model_name

if best_model_path is not None:
    dst_path = os.path.join(trained_model_dir, 'trained_model_best.pkl')
    shutil.copyfile(best_model_path, dst_path)
    print(f"[INFO] Best model saved : epoch {best_epoch} with PSNR {best_psnr:.4f}")

# save onnx
dummy_input = (
    torch.randn(1, 3, 256, 256),
    torch.randn(1, 3, 256, 256),  
    torch.randn(1, 3, 256, 256),
)

onnx_path = f"/home/jaekim/ws/git/myHDR/models_onnx/model-{args.model}.onnx"
torch.onnx.export(
    model.cpu().module, 
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['short', 'mid', 'long'],
    output_names=['output'],
    dynamic_axes={
        'short': {0: 'batch_size', 2: 'height', 3: 'width'},
        'mid': {0: 'batch_size', 2: 'height', 3: 'width'},
        'long': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
)

pt2_path = f"/home/jaekim/ws/git/myHDR/models_onnx/model-{args.model}.pt2"

exported = torch.export.export(
    model.cpu().module,
    dummy_input,
)
torch.export.save(exported, pt2_path)
    
end_train = time.time()

print(f"[INFO] Training finished. ({end_train - start_train:.4f} seconds)")
wandb.finish()
#save_plot(trained_model_dir)
