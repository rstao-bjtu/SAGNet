import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import os

from .datasets import dataset_folder

vals=['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def get_dataset(opt, classes=''):
    dset_lst = []
    # if opt.isTrain:
    #     for cls in os.listdir(arg_root) if multiclass[vals.index(arg_root.split('/')[-1])] else ['']:
    #         root = arg_root + '/' + cls
    #         dset = dataset_folder(opt, root)
    #         dset_lst.append(dset)
    # import pdb; pdb.set_trace()
    if opt.isTrain:
        for cls in classes:
            root = opt.dataroot + '/'+ cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
    else:
        for cls in opt.classes:
            root = opt.dataroot + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt, classes=''):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt, classes)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
