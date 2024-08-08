#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

bt=128
feature_dim=128
fig_token=''
use_train_data=1

case=9
if [ $case -eq 1 ]; then
  feature_dim=128
  lambda=0.005
  ckpt_path='lmbda0.005_128_128_cifar10/0.005_128_128_cifar10_model.pth'
elif [ $case -eq 2 ]; then
  feature_dim=128
  lambda=0.05
  ckpt_path='lmbda0.05_dim128/0.05_128_128_cifar10_model.pth'
elif [ $case -eq 7 ]; then
  lambda=0.005
  feature_dim=128
  ckpt_path='cifar10_linear_feat128_lmbda0.005_lr1e-3_wd1e-6_bt128_sameInit2/model.pth'
  fig_token='_sameInit2'
elif [ $case -eq 8 ]; then
  feature_dim=128
  ckpt_path='cifar10_linear_feat128_lmbda0.005_lr1e-3_wd1e-6_bt128_tmp_model_init.pth'
  fig_token='_randInit'
elif [ $case -eq 9 ]; then
  feature_dim=128
  fig_token='_randInit_noLoading'
fi
ckpt_path='./results/'$ckpt_path
fig_dir='./figs/dim'$feature_dim'_lmbda'$lambda'_bt'$bt$fig_token

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


