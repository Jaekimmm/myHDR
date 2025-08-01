import numpy as np
import time
import argparse
import torch.utils.data

from model import *
from running_func import *
from utils import *

import os

parser = argparse.ArgumentParser(description='Attention-guided HDR')

parser.add_argument('--train_name', type=str, default='')
parser.add_argument('--test_name', type=str, default='')
parser.add_argument('--test_data', default='./dataset_test')

parser.add_argument('--model', type=str, default='LIGHTFUSE')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')

parser.add_argument('--loss', nargs='+', type=str, default='VGG')
parser.add_argument('--loss_weights', nargs='+', type=float, default=1)
parser.add_argument('--epoch', type=int, default=0)

parser.add_argument('--load_gating_map', type=str, default='')
parser.add_argument('--map_preproc', type=str, default='')
parser.add_argument('--map_index_enable', type=int, default=1)
parser.add_argument('--output_gating', type=str, default='none')

parser.add_argument('--offset', action='store_true', default=False)
parser.add_argument('--input_tonemap', type=str, default=None)
parser.add_argument('--label_tonemap', type=str, default=None)
parser.add_argument('--output_tonemap', type=str, default=None)
parser.add_argument('--adjust_exp_to', type=str, default=None)

parser.add_argument('--use_cuda', default=True)
parser.add_argument('--load_model', default=True)

parser.add_argument('--seed', default=1)
parser.add_argument('--vgg_layer', type=int, default=22)
parser.add_argument('--lr', default=0.0001)


args = parser.parse_args()


torch.manual_seed(args.seed)

print("\n\n << CUDA devices >>")
if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    cuda_count = torch.cuda.device_count()
    print(f"Number of visible CUDA devices: {cuda_count}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
else:
    print("CUDA is not available.\n")
    
#load data
testimage_dataset = torch.utils.data.DataLoader(
    data_loader(args.test_data, crop_size=0, geometry_aug=False, map=args.load_gating_map),
    batch_size=1, shuffle=False)

#make folders of trained model and result
trained_model_dir = f"./trained-model-{args.model}-{args.train_name}/"
if args.epoch == 0:
    trained_model_filename = f"trained_model_best.pkl"
else:
    trained_model_filename = f"trained_model{args.epoch}.pkl"
outdir = f"./result-{args.model}-{args.train_name}-{args.test_name}/"
mk_dir(outdir)


# Get model
if args.model in globals(): 
    model = globals()[args.model]
model = model(args)
model.apply(weights_init_kaiming)
 
if args.use_cuda:
    model.cuda()
    model = nn.DataParallel(model)

print(f"\nRun test with model {model}\n")
print(model)

print(f"[INFO] Start test with model {model}")
print(f"[INFO] Input preprocessing : Offset {args.offset}, Input tonemap {args.input_tonemap}, Label tonemap {args.label_tonemap}")

##
model = model_load(model, trained_model_dir, trained_model_filename)
loss_func = get_loss_function(args)

if args.load_gating_map != '':
    gate_func = get_gating_scheme('load_map')
else:
    gate_func = get_gating_scheme(args.output_gating)

start = time.time()
loss = testing_gate(model, loss_func, gate_func, testimage_dataset, outdir, args)
end = time.time()
