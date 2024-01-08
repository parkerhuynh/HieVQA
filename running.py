import argparse
import os
import math
import numpy as np
import random
import time

import json
from pathlib import Path
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from utils import *
from dataset import create_dataset, create_sampler, create_loader
from optim import create_optimizer  
from models import get_model
from scheduler import create_scheduler
from torch import nn
import pandas as pd
import wandb
from datetime import datetime
from loss import HierarchicalLoss

from engines import trainer, validator
import warnings
import torch.distributed as dist
# Hiding runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)



def main(args):
    print()
    # Initialize the distributed computing environment and other settings
    device, world_size = setup_environment(args)
    if is_main_process() and args.save_model:
        args.output_dir = output_path_create(args)
    if args.bs > 0:
        args.batch_size_train = int(float(args.bs)/world_size)
    
    if args.bs_test > 0:
        args.batch_size_test = int(float(args.bs_test)/world_size)
    # Initialize WandB for experiment tracking, if enabled.
    if args.wandb:
        initialize_wandb(args)
    
    # Create datasets for training, validation, or testing.
    # Handles both distributed and non-distributed data handling.
    if args.wandb and is_main_process():
        directory = os.getcwd()
        file_list = list_files_and_subdirectories(directory, args)
        for filename in file_list:
            file_path = os.path.join(directory, filename)
            wandb.save(file_path, directory)
        

    print("\n-----------CREATING DATASETS-------------\n")
    if args.checkpoint != "":
        print("Not Implement")
        
        # Create and evaluate the test dataset
        # if is_main_process():
        #     test_dataset = create_dataset(args, istrain=False)
            
        #     if args.distributed:
        #         num_tasks = get_world_size()
        #         global_rank = get_rank()
        #         samplers = create_sampler(test_dataset, [False], num_tasks, global_rank)
        #     else:
        #         samplers = [None]
        #     test_loader = create_loader(test_dataset, samplers, args, istrain =False)

        #     # Fetch and set up the model according to specified configurations.
        #     model  = torch.load(os.path.join(args.checkpoint))
        #     model = model.to(device)
        #     result = evaluator(model, test_loader, device)
        #     result.to_csv(os.path.join("/".join(args.checkpoint.split("/")[:-1]), "result.csv"), index=False)
    else:
        # Create training and validation datasets
        print('> Creating training and validation sets'.upper())
        train_dataset, val_dataset = create_dataset(args)
        print(f"Data Size:\n  Train: {len(train_dataset)}\n  Validation:{len(val_dataset)}")
        datasets = [train_dataset, val_dataset]
        #DISTRIBUTE DATASET
        if args.distributed:
            num_tasks = get_world_size()
            global_rank = get_rank()
            samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
        else:
            samplers = [None, None]
        
        train_dataset_size = len(train_dataset)

        #CREATE DATASET LOADERS
        train_loader, val_loader = create_loader(datasets, samplers, args)
        args.optimizer['lr'] = float(args.optimizer['lr'])
        args.schedular['lr'] = float(args.schedular['lr'])
        model = get_model(args, train_dataset)
        model = model.to(device)
        
        #OPTIMIZER and LOSS
        arg_opt = AttrDict(args.optimizer)
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = AttrDict(args.schedular)
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (args.batch_size_train * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)
        loss_fn =  HierarchicalLoss(args, train_dataset)

        # lr_scheduler = None
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        #SET UP
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True).to(device)
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        max_epoch = args.schedular['epochs']
        if args.debug:
            max_epoch = 100
            val_loader = train_loader
        start_epoch = 0
        best_loss = 10000
        if is_main_process():
            print(model)
        for epoch in range(start_epoch, max_epoch):
            print("\n\n" + "--"*50 + f" Epoch {epoch} " + "--"*50 + "\n")
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = trainer(model, train_loader, optimizer, loss_fn, epoch, device, lr_scheduler, args, wandb)
            validation_stats = validator(model, val_loader, device, loss_fn, args, train_dataset.idx_to_answer_type)

            if args.wandb:
                wandb_train_log = {**{f'train_{k}': float(v) for k, v in train_stats.items()},
                                'epoch': epoch}
                wandb_val_log = {**{f'val_{k}': float(v) for k, v in validation_stats.items()}}
                wandb_train_log.update(wandb_val_log)
                wandb.log(wandb_train_log)


            if is_main_process() and args.save_model:
                if hasattr(model, 'module'):
                    model_without_ddp = model.module
                #Save model
                last_model_path = os.path.join(args.output_dir, "model_latest_epoch.pt")
                torch.save(model_without_ddp, last_model_path)
                
                if validation_stats['weighted_loss'] < best_loss:
                    best_loss = validation_stats['weighted_loss']
                    best_model_path = os.path.join(args.output_dir, "best_model_state.pt")
                    torch.save(model_without_ddp, best_model_path)
            print("\n")
        if is_main_process():
            wandb.finish()
            

        
                    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int)    
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--bs', default=0, type=int)
    parser.add_argument('--bs_test', default=0, type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--note', type=str)
    parser.add_argument('--version', type=str)
    parser.add_argument('--task', type=str, required=True, choices=['cvqa', 'qrvqa','acvqa', 'mergevqa', 'hievqa'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_dir', type=str, help="for fine-tuning")
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--save_model', action='store_true')

    args = parser.parse_args()

    #Check data path
    print("\n-----------CONFIGS-------------\n")
    if not os.path.exists(args.data_path):
        print(args.data_path)
        raise FileNotFoundError(f"The data path {args.data_path} does not exist")
    
    #Load Config
    model_config, data_config, training_config = load_configuration(args.model, args.dataset)
    args.model_config = model_config
    args.data_config = data_config
    vars(args).update(training_config)
    print_namespace_as_table(args)
    main(args)

    