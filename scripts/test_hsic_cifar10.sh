#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

bt=128
fig_token=''

case=1
if [ $case -eq 1 ]; then
  lambda=0.005
  feature_dim=512
  ckpt_path='neg_corr_0.005_512_128_cifar10_model.pth'
elif [ $case -eq 2 ]; then
  lambda=0.05
  feature_dim=512
  ckpt_path='neg_corr_0.05_512_128_cifar10_model.pth'
fi
ckpt_path='./results/'$ckpt_path
fig_dir='./figs/hsic_dim'$feature_dim'_lmbda'$lambda'_bt'$bt$fig_token
fSinVals=$fig_dir'/sinVals'

# the test bt should be no smaller than the feat dim
# in order for the check on singular values to be meaningful.
test_bt=$feature_dim

WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=1 \
  python main.py \
    --corr_neg_one \
    --test-only=1 \
    --dataset=$dataset \
    --feature_dim=$feature_dim \
    --proj-head-type=$proj_head_type \
    --batch_size=$test_bt \
    --load-ckpt=1 \
    --pretrained-path=$ckpt_path \
    --fSinVals=$fSinVals

