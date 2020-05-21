from __future__ import print_function

import argparse
import itertools
import os
import pickle
import sys
from datetime import datetime

import matplotlib
import numpy as np
import torch
import torchvision

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import code.archs as archs
from code.utils.cluster.general import config_to_str, get_opt, update_lr, nice
from code.utils.cluster.data import cluster_twohead_create_dataloaders
from code.utils.cluster.cluster_eval import cluster_eval, get_subhead_using_loss
from code.utils.cluster.IID_losses import IID_loss
from code.utils.cluster.render import save_progress

"""
  Fully unsupervised clustering ("IIC" = "IID").
  Train and test script (greyscale datasets).
  Network has two heads, for overclustering and final clustering.
"""

# Options ----------------------------------------------------------------------

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

config = parser.parse_args()

# Setup ------------------------------------------------------------------------

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

assert ("MNIST" == config.dataset)                                      # [assert] dataset name
dataset_class = torchvision.datasets.MNIST                              # dataset_class = MNIST?
config.train_partitions = [True, False]
config.mapping_assignment_partitions = [True, False]
config.mapping_test_partitions = [True, False]

if not os.path.exists(config.out_dir):                                  # [assert] output path directory
    os.makedirs(config.out_dir)

    print("Config: %s" % config_to_str(config))


# Model ------------------------------------------------------------------------
def train(render_count=-1):
#     tf1 = RandomCrop or CenterCrop sz:20,    
#         Resize to 24, 
#         ToTensor
#     tf2 = RandomCrop sz: randomly choice (16 or 20 or 24),  
#             Resize to [24, 24],
#             ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.125)
#             ToTensor
#     tf3 = CenterCrop sz:20, 
#             Resize to 24
#             ToTensor

    # Dataloader for train
    # dataloaders_head_A = dataloaders_head_B = [train_dataloader(use tf1, target:MNIST), num_dataloader x train_tf_dataloader(use tf2, target:MNIST)]
    # Dataloader for validate
    # mapping_assignment_dataloader = mapping_test_dataloader = [dataloader(uset tf3, target:MNIST)]
    dataloaders_head_A, dataloaders_head_B, \
    mapping_assignment_dataloader, mapping_test_dataloader = \
    cluster_twohead_create_dataloaders(config)

    net = archs.__dict__[config.arch](config)
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.train()

    optimiser = get_opt(config.opt)(net.module.parameters(), lr=config.lr)

    heads = ["B", "A"]
    if config.head_A_first:
        heads = ["A", "B"]

    head_epochs = {}
    head_epochs["A"] = config.head_A_epochs
    head_epochs["B"] = config.head_B_epochs

    # Results
    # ----------------------------------------------------------------------
    config.epoch_acc = []
    config.epoch_avg_subhead_acc = []
    config.epoch_stats = []

    if config.double_eval:
        config.double_eval_acc = []
        config.double_eval_avg_subhead_acc = []
        config.double_eval_stats = []

    config.epoch_loss_head_A = []
    config.epoch_loss_head_B = []

    sub_head = None
    if config.select_sub_head_on_loss:
        sub_head = get_subhead_using_loss(config, dataloaders_head_B, net,
                                        sobel=False)
        
    _ = cluster_eval(config, net,
                 mapping_assignment_dataloader=mapping_assignment_dataloader,
                 mapping_test_dataloader=mapping_test_dataloader,
                 sobel=False,
                 use_sub_head=sub_head)

    print(
      "Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
    
    if config.double_eval:
        print("double eval: \n %s" % (nice(config.double_eval_stats[-1])))
        sys.stdout.flush()
        next_epoch = 1

    fig, axarr = plt.subplots(6 + 2 * int(config.double_eval), sharex=False,
                            figsize=(20, 20))

    # Train
    # ------------------------------------------------------------------------
    
    # loop for each epoch
    for e_i in xrange(next_epoch, config.num_epochs):
        print("Starting e_i: %d" % e_i)

        # lr reduce when we set parameter lr_schedule
        if e_i in config.lr_schedule:
            optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

        # head_A, head_B
        for head_i in range(2):
            head = heads[head_i]
            if head == "A":
                dataloaders = dataloaders_head_A
                epoch_loss = config.epoch_loss_head_A
            elif head == "B":
                dataloaders = dataloaders_head_B
                epoch_loss = config.epoch_loss_head_B

            avg_loss = 0.  # over heads and head_epochs (and sub_heads)
            avg_loss_count = 0

            # loop for epoch of head_i
            for head_i_epoch in range(head_epochs[head]):
                sys.stdout.flush()

                # dataloaders = [loader_tf1, num_dataloaders * loader_tf2 ]
                # iterator for each dataloader
                iterators = (d for d in dataloaders)   
                
                

                # data batch
                b_i = 0
                for tup in itertools.izip(*iterators):
                    # tup[0][0] : x (img)
                    # tup[0][1] : y (ground truth)
                    
                    net.module.zero_grad()

                    # buffer for all images
                    all_imgs = torch.zeros((config.batch_sz, config.in_channels,
                                          config.input_sz,
                                          config.input_sz)).cuda()
                    all_imgs_tf = torch.zeros((config.batch_sz, config.in_channels,
                                             config.input_sz,
                                             config.input_sz)).cuda()

                    # if config.batch_sz = 700,  num_dataloader = 5
                    # curr_batch_sz = 700/5 = 140
                    imgs_curr = tup[0][0]  # always the first.   img of loader_tf1, shape:(curr_batch_sz, 1(gray), img_size, img_size)     i.e (140, 1, 24, 24)
                    # gt_curr = tup[0][1]  #                     gt  of loader_tf1, shape:(curr_batch_sz, )                                i.e (140, )
                    
                    curr_batch_sz = imgs_curr.size(0)
                    
                    for d_i in xrange(config.num_dataloaders):
                        imgs_tf_curr = tup[1 + d_i][0]  # from 2nd to last
                        assert (curr_batch_sz == imgs_tf_curr.size(0))

                        # save to buffer
                        actual_batch_start = d_i * curr_batch_sz
                        actual_batch_end = actual_batch_start + curr_batch_sz
                        all_imgs[actual_batch_start:actual_batch_end, :, :, :] = \
                          imgs_curr.cuda()
                        all_imgs_tf[actual_batch_start:actual_batch_end, :, :, :] = \
                          imgs_tf_curr.cuda()

                    if not (curr_batch_sz == config.dataloader_batch_sz):
                        print("last batch sz %d" % curr_batch_sz)

                    curr_total_batch_sz = curr_batch_sz * config.num_dataloaders
                    
                    # times 2
                    all_imgs = all_imgs[:curr_total_batch_sz, :, :, :]        # i.e (700, 1, 24, 24)
                    all_imgs_tf = all_imgs_tf[:curr_total_batch_sz, :, :, :]  # i.e (700, 1, 24, 24)

                    # forawrd
                    x_outs = net(all_imgs)
                    x_tf_outs = net(all_imgs_tf)

                    avg_loss_batch = None  # avg over the heads
                    for i in xrange(config.num_sub_heads):
                        loss = IID_loss(x_outs[i], x_tf_outs[i])
                        if avg_loss_batch is None:
                            avg_loss_batch = loss
                        else:
                            avg_loss_batch += loss

                    # average of loss
                    avg_loss_batch /= config.num_sub_heads             

                    if ((b_i % 100) == 0) or (e_i == next_epoch):
                        print(
                          "Model ind %d epoch %d head %s batch: %d avg loss %f time %s" % \
                          (config.model_ind, e_i, head, b_i, avg_loss_batch.item(), datetime.now()))
                        sys.stdout.flush()

                    # exit when loss has infinity value
                    if not np.isfinite(avg_loss_batch.item()):
                        print("Loss is not finite... %s:" % avg_loss_batch.item())
                        exit(1)

                    avg_loss += avg_loss_batch.item()
                    avg_loss_count += 1
                    
                    avg_loss_batch.backward()
                    optimiser.step()

                    b_i += 1
                    if b_i == 2 and config.test_code:
                        break

            avg_loss = float(avg_loss / avg_loss_count)

            # logging
            epoch_loss.append(avg_loss)

        # Eval
        # -----------------------------------------------------------------------

        sub_head = None
#         if config.select_sub_head_on_loss:
#             sub_head = get_subhead_using_loss(config, dataloaders_head_B, net,
#                                             sobel=False, lamb=config.lamb_B)

        # True or False
        is_best = cluster_eval(config, net,
                               mapping_assignment_dataloader=mapping_assignment_dataloader,
                               mapping_test_dataloader=mapping_test_dataloader,
                               sobel=False,
                               use_sub_head=sub_head)

        print(
          "Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
        if config.double_eval:
            print("double eval: \n %s" % (nice(config.double_eval_stats[-1])))
        sys.stdout.flush()

        axarr[0].clear()
        axarr[0].plot(config.epoch_acc)
        axarr[0].set_title("acc (best), top: %f" % max(config.epoch_acc))

        axarr[1].clear()
        axarr[1].plot(config.epoch_avg_subhead_acc)
        axarr[1].set_title("acc (avg), top: %f" % max(config.epoch_avg_subhead_acc))

        axarr[2].clear()
        axarr[2].plot(config.epoch_loss_head_A)
        axarr[2].set_title("Loss head A")

        axarr[3].clear()

        axarr[4].clear()
        axarr[4].plot(config.epoch_loss_head_B)
        axarr[4].set_title("Loss head B")

        axarr[5].clear()

        if config.double_eval:
            axarr[6].clear()
            axarr[6].plot(config.double_eval_acc)
            axarr[6].set_title("double eval acc (best), top: %f" %
                             max(config.double_eval_acc))

            axarr[7].clear()
            axarr[7].plot(config.double_eval_avg_subhead_acc)
            axarr[7].set_title("double eval acc (avg)), top: %f" %
                             max(config.double_eval_avg_subhead_acc))

        fig.tight_layout()
        fig.canvas.draw_idle()
        fig.savefig(os.path.join(config.out_dir, "plots.png"))

        if is_best or (e_i % config.save_freq == 0):
            net.module.cpu()

            if e_i % config.save_freq == 0:
            torch.save(net.module.state_dict(),
                       os.path.join(config.out_dir, "latest_net.pytorch"))
            torch.save(optimiser.state_dict(),
                       os.path.join(config.out_dir, "latest_optimiser.pytorch"))

            config.last_epoch = e_i  # for last saved version

            if is_best:
                # also serves as backup if hardware fails - less likely to hit this
                torch.save(net.module.state_dict(),
                           os.path.join(config.out_dir, "best_net.pytorch"))
                torch.save(optimiser.state_dict(),
                           os.path.join(config.out_dir, "best_optimiser.pytorch"))

                with open(os.path.join(config.out_dir, "best_config.pickle"),
                          'wb') as outfile:
                    pickle.dump(config, outfile)

                with open(os.path.join(config.out_dir, "best_config.txt"),
                          "w") as text_file:
                    text_file.write("%s" % config)

            net.module.cuda()

        with open(os.path.join(config.out_dir, "config.pickle"),
                  'wb') as outfile:
            pickle.dump(config, outfile)

        with open(os.path.join(config.out_dir, "config.txt"),
                  "w") as text_file:
            text_file.write("%s" % config)

        if config.test_code:
            exit(0)

train()