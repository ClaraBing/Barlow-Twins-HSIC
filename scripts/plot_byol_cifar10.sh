#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

bt=128
feature_dim=128
fig_token=''
use_train_data=1

case=1
if [ $case -eq 1 ]; then
  feature_dim=128
  lambda=0.005
  ckpt_path='byol_128_512_512_cifar10_model_1000.pth'
fi
ckpt_path='./results/'$ckpt_path
fig_dir='./figs/byol_dim'$feature_dim'_lmbda'$lambda'_bt'$bt$fig_token

WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=0 \
  python plot.py \
    --dataset=$dataset \
    --feature_dim=$feature_dim \
    --proj-head-type=$proj_head_type \
    --batch_size=$bt \
    --ckpt_path=$ckpt_path \
    --use-train-data=$use_train_data \
    --fig-dir=$fig_dir


