#!/bin/bash

# Constants for Distributed Training Configuration
NNODES=1
MASTER_ADDR='localhost'
MASTER_PORT=7778
NPROC_PER_NODE=4 # Number of processes per node
# Variables for the command
data_path="/home/reda/scratch/ngoc/data/simpsonsvqa"
dataset="simpsonsvqa"
model="VQA"
output_dir="/home/reda/scratch/ngoc/code/HieVQA/results"
task=hie-vqa-wo-unans # vqa-w-ans & vqa-hie
note="Hie-VQA-WO-Unans"
version="v2"
bs=256
bs_test=256


# Print system information
echo "=========== System Information ==========="
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MASTER_PORT: $MASTER_PORT"
echo "OUT PATH: $output_dir"
echo "WORD SIZE: $word_size"
echo "DATA PATH: $data_path"
echo "DATASET: $dataset"
echo "MODEL: $model"
echo "TASK: $task"
echo "NOTE: $note"
# Run the command
CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=$NPROC_PER_NODE python3 -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --use_env running.py \
    --data_path "$data_path" \
    --model "$model" \
    --output_dir "$output_dir" \
    --task "$task" \
    --note "$note" \
    --dataset "$dataset" \
    --bs "$bs" \
    --bs_test "$bs_test" \
    --version "$version" \
    --wandb