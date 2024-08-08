#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

bt=128
test_bt=128

save_token='2'

case=1
if [ $case -eq 1 ]; then
  lambda=0.005
  feature_dim=128
  ckpt_path='byol_128_512_512_cifar10_model_1000.pth'
fi
ckpt_path='./results/'$ckpt_path
fig_dir='./figs/byol_dim'$feature_dim'_lmbda'$lambda'_bt'$bt$save_token
fSinVals=$fig_dir'/sinVals'
save_feats=1
fsave_feats='./saved_feats/byol_dim'$feature_dim'_lmbda'$lambda'_bt'$bt$save_token

WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=0 \
  python main.py \
    --test-only=1 \
    --dataset=$dataset \
    --feature_dim=$feature_dim \
    --proj-head-type=$proj_head_type \
    --batch_size=$test_bt \
    --load-ckpt=1 \
    --pretrained-path=$ckpt_path \
    --fSinVals=$fSinVals \
    --save-feats=$save_feats \
    --fsave-feats=$fsave_feats

