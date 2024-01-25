import sys
import time
import os
import csv
import torch
from util import Logger
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from models import get_model



vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]
# Running tests
opt = TestOptions().parse(print_options=False)

# opt.dataroot = '/opt/data/private/tcc/data/data/CNNDetection/test/'
opt.model_path = 'checkpoints/clip_chair_horse_2_domains_4_classes_baseline__2023_12_26_17_40_07__lnum_64__random_sobel/{}'.format(opt.pth_dataroot.split('/')[-1])
opt.loadSize = 224
opt.batch_size = 32
model_name = os.path.basename(opt.model_path).replace('.pth', '')

results_dir = './results_onprogan/'
logpath = os.path.join(results_dir, opt.model_path.split('/')[-2])
os.makedirs(results_dir, mode = 0o777, exist_ok = True) 
os.makedirs(logpath, mode = 0o777, exist_ok = True) 
Logger(os.path.join(logpath, opt.model_path.split('/')[-1] + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+'.log'))

dataroot = opt.dataroot
print(f'Dataroot {opt.dataroot}')
print(f'Model_path {opt.model_path}')

accs = [];aps = []
print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

model = get_model('CLIP:ViT-L/14')
from collections import OrderedDict
from copy import deepcopy
state_dict = torch.load(opt.model_path, map_location='cpu')['model']
net_params = sum(map(lambda x: x.numel(), model.parameters()))
print(f'Model parameters {net_params:,d}')
pretrained_dict = OrderedDict()
for ki in state_dict.keys():
    pretrained_dict[ki[7:]] = deepcopy(state_dict[ki])
model.load_state_dict(pretrained_dict, strict=True)
model.cuda()
model.eval()

pretrained_dict = OrderedDict()
for ki in state_dict.keys():
    pretrained_dict[ki[7:]] = deepcopy(state_dict[ki])

model.load_state_dict(pretrained_dict, strict=True)

model.cuda()
model.eval()

for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = False    # testing without resizing by default
    opt.no_crop = True    # testing without resizing by default

    acc, ap, _, _, _, _ = validate(model, opt)
    accs.append(acc);aps.append(ap)
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

