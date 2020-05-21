import argparse
import itertools
import os
import pickle
import sys
import numpy as np
import torch
import torchvision
from torchvision import transforms

import code.archs as archs


########################## argument ########################
parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)             # model id
parser.add_argument("--arch", type=str, default="ClusterNet4h")
parser.add_argument("--opt", type=str, default="Adam")                  # optimizer
parser.add_argument("--mode", type=str, default="IID")

parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--dataset_root", type=str,
                    default="/scratch/local/ssd/xuji/MNIST")

parser.add_argument("--gt_k", type=int, default=10)                      # command -> 10
parser.add_argument("--output_k_A", type=int, required=True)             # command -> 50
parser.add_argument("--output_k_B", type=int, required=True)             # command -> 10

parser.add_argument("--lr", type=float, default=0.01)                    # command -> 0.0001    initial learning rate
parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])    # epoch schedule to reduce lr 
parser.add_argument("--lr_mult", type=float, default=0.1)                # rate to reduce lr 

parser.add_argument("--num_epochs", type=int, default=1000)              # command -> 3200      epoch
parser.add_argument("--batch_sz", type=int, default=240)  # num pairs    # command -> 700       batch size
parser.add_argument("--num_dataloaders", type=int, default=3)            # command -> 5
parser.add_argument("--num_sub_heads", type=int, default=5)              # command -> 5

parser.add_argument("--out_root", type=str,                     
                    default="/scratch/shared/slow/xuji/iid_private")     # output root path
parser.add_argument("--restart", dest="restart", default=False,
                    action="store_true")
parser.add_argument("--restart_from_best", dest="restart_from_best",
                    default=False, action="store_true")
parser.add_argument("--test_code", dest="test_code", default=False,
                    action="store_true")

parser.add_argument("--save_freq", type=int, default=20)

parser.add_argument("--double_eval", default=False, action="store_true")

parser.add_argument("--head_A_first", default=False, action="store_true")
parser.add_argument("--head_A_epochs", type=int, default=1)
parser.add_argument("--head_B_epochs", type=int, default=1)                  # command -> 2

parser.add_argument("--batchnorm_track", default=False, action="store_true")

parser.add_argument("--select_sub_head_on_loss", default=False,
                    action="store_true")

# transforms
parser.add_argument("--demean", dest="demean", default=False,
                    action="store_true")
parser.add_argument("--per_img_demean", dest="per_img_demean", default=False,
                    action="store_true")
parser.add_argument("--data_mean", type=float, nargs="+",
                    default=[0.5, 0.5, 0.5])
parser.add_argument("--data_std", type=float, nargs="+",
                    default=[0.5, 0.5, 0.5])

parser.add_argument("--crop_orig", dest="crop_orig", default=False,            # command -> True
                    action="store_true")
parser.add_argument("--crop_other", dest="crop_other", default=False,          # command -> True
                    action="store_true")
parser.add_argument("--tf1_crop", type=str, default="random")                  # command -> centre_half
parser.add_argument("--tf2_crop", type=str, default="random")                  # command -> random
parser.add_argument("--tf1_crop_sz", type=int, default=84)                     # command -> 20
parser.add_argument("--tf2_crop_szs", type=int, nargs="+",                     # command -> 16 20 24
                    default=[84])  # allow diff crop for imgs_tf
parser.add_argument("--tf3_crop_diff", dest="tf3_crop_diff", default=False,
                    action="store_true")
parser.add_argument("--tf3_crop_sz", type=int, default=0)
parser.add_argument("--input_sz", type=int, default=96)                        # command -> 24

parser.add_argument("--no_jitter", dest="no_jitter", default=False,
                    action="store_true")
parser.add_argument("--no_flip", dest="no_flip", default=False,                # command -> True
                    action="store_true")

parser.add_argument("--model_load", dest="model_load", type=str, default="")
config = parser.parse_args()
###############################################################################


#################### SETUP #######################


config.twohead = True
config.in_channels = 1
config.out_dir = os.path.join(config.out_root, str(config.model_ind))   # output path
assert (config.batch_sz % config.num_dataloaders == 0)                  # [assert] batch size % num data lodaer  = 0
config.dataloader_batch_sz = config.batch_sz / config.num_dataloaders   # batch size of data loader

assert (config.mode == "IID")                                           # [assert]
assert ("TwoHead" in config.arch)
assert (config.output_k_B == config.gt_k)
config.output_k = config.output_k_B  # for eval code
assert (config.output_k_A >= config.gt_k)
config.eval_mode = "hung"

# assert ("MNIST" == config.dataset)                                      # [assert] dataset name
# dataset_class = torchvision.datasets.MNIST                              # dataset_class = MNIST?
# config.train_partitions = [True, False]
# config.mapping_assignment_partitions = [True, False]
# config.mapping_test_partitions = [True, False]

#######################################################



id_arch = 'ClusterNet6cTwoHead'
print(archs.__dict__)[id_arch]
net = archs.__dict__[id_arch](config)

print(config.model_load)
net.load_state_dict(torch.load(config.model_load))

batch_size = 500

imgnet_data = torchvision.datasets.MNIST(config.dataset_root, 
                                         train=False,
                                        transform=transforms.Compose([transforms.ToTensor()])
                                        )
dataloader = torch.utils.data.DataLoader(imgnet_data,
                                       batch_size=batch_size,
                                       # full batch
                                       shuffle=True)

print(type(imgnet_data), imgnet_data)
print('\n')
print(imgnet_data.__len__)
print('\n')
print(net)

decode = {0:4, 1:2, 2:9, 3:6, 4:7, 5:3, 6:1, 7:0, 8:5, 9:8}
best_subnet = 3

n_total = 0
n_correct = 0
# for xx, yy in dataloader:
#     #print(type(xx), xx.shape)
#     # print(type(yy), yy.shape, yy, yy[0])
    
#     pred = net(xx)
#     #print(type(pred), len(pred), pred[0].shape)
    
#     for b_i in range(batch_size):
#         gt = yy[b_i].item()
#         pred_ix = pred[3][b_i].max(0)[1].item()
#         pred_class = decode[pred_ix]
        
#         print(gt, pred_class)
        
#         if gt == pred_class:
#             n_correct += 1
#     n_total += batch_size
#     print(n_total)

print(n_correct, n_total)
print('done')