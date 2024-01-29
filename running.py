import argparse
import os
import math
import numpy as np
import random
import time
from metrics import calculate_accuracies
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
from transformers import AdamW
from engines import trainer, validator
import warnings
import torch.distributed as dist
# Hiding runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO
from PIL import Image
import numpy as np
import torch.onnx
import glob
from loss import HierarchicalLoss  
def main(args):
    print()
    init_distributed_mode(args)
    device = torch.device(args.device)
    world_size = get_world_size()
    
    if args.bs > 0:
        args.batch_size_train = int(float(args.bs)/world_size)
    
    if args.bs_test > 0:
        args.batch_size_test = int(float(args.bs_test)/world_size)
        
    seed = args.seed + get_rank()
    print("#"*100)
    print(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    
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
        optimizer = AdamW(model.parameters(), lr=1e-5)
        loss_fn =  HierarchicalLoss(args, train_dataset)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True).to(device)
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        
        max_epoch = args.schedular['epochs']
        start_epoch = 0
        best_acc = 0
        val_prediction_csv = None
        if is_main_process():
            print(model)
        val_loader = train_loader
        for epoch in range(start_epoch, max_epoch):
            print("\n\n" + "--"*50 + f" Epoch {epoch} " + "--"*50 + "\n")
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = trainer(model, train_loader, optimizer, loss_fn, epoch, device, args, wandb)
            
            validation_stats, val_prediction = validator(model, val_loader, device, loss_fn, args, epoch)
            combine_result = collect_result(val_prediction, f'vqa_result_{args.model}_{args.task}_{args.dataset}_epoch{epoch}', local_wdir="./temp_result")
            if args.wandb:
                wandb_train_log = {**{f'train_{k}': float(v) for k, v in train_stats.items()},
                                'epoch': epoch}
                wandb_val_log = {**{f'val_{k}': float(v) for k, v in validation_stats.items()}}
                wandb_train_log.update(wandb_val_log)
                wandb.log(wandb_train_log)


            if is_main_process() and args.wandb:
                
                combine_result_csv = pd.DataFrame(combine_result)
                val_accuracies, val_prediction_csv_i = calculate_accuracies(combine_result_csv, val_dataset)
                
                # wandb_val_log = {**{f'val_{k}': float(v) for k, v in val_accuracies.items()}}
                val_accuracies.update({
                    "epoch": epoch
                })
                wandb.log(val_accuracies)
                if hasattr(model, 'module'):
                    model_without_ddp = model.module
                #Save model
                print(val_accuracies)
                wandb.log(val_accuracies)
                last_model_path = os.path.join(args.output_dir, "model_latest_epoch.pt")
                torch.save(model_without_ddp, last_model_path)
                
                if val_accuracies['val_accuracy_vqa(vqa-wo-unans)'] > best_acc:
                    val_prediction_csv = val_prediction_csv_i
                    best_acc = val_accuracies['val_accuracy_vqa(vqa-wo-unans)']
                    wandb_log_val_accuracy_best = {**{f'best_{k}': v for k, v in val_accuracies.items()}}
                    wandb.log(wandb_log_val_accuracy_best)
                    
                    best_model_path = os.path.join(args.output_dir, "best_model_state.pt")
                    torch.save(model_without_ddp, best_model_path)
                    
            print("\n")
        if is_main_process() and args.wandb:

            val_prediction_csv.to_csv("prediction.csv", index=False)
            val_prediction_csv = val_prediction_csv.sort_values(by='question_id')
            directory = os.getcwd()
            file_path = os.path.join(directory, "prediction.csv")
            wandb.save(file_path, directory)
            
            y_true = val_prediction_csv['small_answer_type_target']
            y_pred = val_prediction_csv['small_answer_type_prediction']
            conf_matrix = confusion_matrix(y_true, y_pred, labels=y_true.unique())
            
            # Normalize the confusion matrix
            conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            
            # Set up the matplotlib figure for side-by-side plots
            plt.figure(figsize=(30, 12))
            
            # Regular confusion matrix
            plt.subplot(1, 2, 1)
            sns.heatmap(conf_matrix, annot=True, fmt='d', 
                        xticklabels=y_true.unique(), yticklabels=y_true.unique(), 
                        cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Type')
            plt.ylabel('True Type')
            
            # Normalized confusion matrix
            plt.subplot(1, 2, 2)
            sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', 
                        xticklabels=y_true.unique(), yticklabels=y_true.unique(), 
                        cmap='Blues')
            plt.title('Normalized Confusion Matrix')
            plt.xlabel('Predicted Type')
            plt.ylabel('True Type')
            
            # Main title
            plt.suptitle(f'{args.task}-{args.version}-{args.version}-{args.dataset}-{args.task}-{args.created}')
            
            # Save the plot to a buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image = Image.open(buffer)
            image_array = np.array(image)
            
            # Log the image to wandb
            wandb.log({"Confusion Matrix": wandb.Image(image_array)})
            
            # Close the plot
            plt.close()
            
            y_true = val_prediction_csv['answer_type']
            y_pred = val_prediction_csv['answer_type_prediction']
            conf_matrix = confusion_matrix(y_true, y_pred, labels=y_true.unique())
            
            # Normalize the confusion matrix
            conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            
            # Set up the matplotlib figure for side-by-side plots
            plt.figure(figsize=(30, 12))
            
            # Regular confusion matrix
            plt.subplot(1, 2, 1)
            sns.heatmap(conf_matrix, annot=True, fmt='d', 
                        xticklabels=y_true.unique(), yticklabels=y_true.unique(), 
                        cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Type')
            plt.ylabel('True Type')
            
            # Normalized confusion matrix
            plt.subplot(1, 2, 2)
            sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', 
                        xticklabels=y_true.unique(), yticklabels=y_true.unique(), 
                        cmap='Blues')
            plt.title('Normalized Confusion Matrix')
            plt.xlabel('Predicted Type')
            plt.ylabel('True Type')
            
            # Main title
            plt.suptitle(f'{args.task}-{args.version}-{args.version}-{args.dataset}-{args.task}-{args.created}')
            
            # Save the plot to a buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image = Image.open(buffer)
            image_array = np.array(image)
            
            # Log the image to wandb
            wandb.log({"Confusion Matrix (3 types)": wandb.Image(image_array)})
            
            # Close the plot
            plt.close()
                    
            directory = os.getcwd()
            file_path = os.path.join(directory, f"model_{args.code_version}.onnx")
            wandb.save(file_path, directory)

            


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
    parser.add_argument('--task', type=str, required=True, choices=['vqa-wo-unans', 'vqa-w-unans', 'hie-vqa-wo-unans-9types'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_dir', type=str, help="for fine-tuning")
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--seed', default=42, type=int)
    
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
    
    pattern = os.path.join("./temp_result", f"vqa_result_{args.model}_{args.task}_{args.dataset}_epoch_*")
    files_to_remove = glob.glob(pattern)

    # Iterate and remove each file
    for file_path in files_to_remove:
        if os.path.isfile(file_path):  # Check if it's a file
            os.remove(file_path)