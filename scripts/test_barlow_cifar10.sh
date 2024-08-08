#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

bt=128

save_token=''

case=10
if [ $case -eq 1 ]; then
  lambda=0.005
  feature_dim=128
  ckpt_path='0.005_128_128_cifar10_model.pth'
elif [ $case -eq 2 ]; then
  lambda=0.05
  feature_dim=128
  ckpt_path='0.05_128_128_cifar10_model.pth'
elif [ $case -eq 3 ]; then
  lambda=0.05
  feature_dim=256
  # last ckpt
  ckpt_path='0.05_256_128_cifar10_model.pth'
  # ckpt in the middle
  ckpt_path='0.05_256_128_cifar10_model_150.pth'
elif [ $case -eq 4 ]; then
  lambda=0.05
  feature_dim=512
  ckpt_path='0.05_512_128_cifar10_model.pth'
elif [ $case -eq 5 ]; then
  lambda=0.005
  feature_dim=128
  ckpt_path='0.005_128_128_cifar10_model.pth'
elif [ $case -eq 6 ]; then
  lambda=0.005
  feature_dim=128
  ckpt_path='cifar10_linear_feat128_lmbda0.005_lr1e-4_wd1e-6_bt128_noNormStd_model.pth'
elif [ $case -eq 7 ]; then
  lambda=0.005
  feature_dim=128
  ckpt_path='cifar10_linear_feat128_lmbda0.005_lr1e-3_wd1e-6_bt128_sameInit2/model.pth'
  save_token='_sameInit2'
elif [ $case -eq 8 ]; then
  feature_dim=128
  ckpt_path='cifar10_linear_feat128_lmbda0.005_lr1e-3_wd1e-6_bt128_tmp_model_init.pth'
  save_token='_randInit'
elif [ $case -eq 9 ]; then
  feature_dim=128
  save_token='_randInit_noLoading'
elif [ $case -eq 10 ]; then
  lambda=0.05
  feature_dim=128
  ckpt_path='cifar10_linear_feat128_lmbda0.05_lr1e-3_wd1e-6_bt128_diffInit/model.pth'
  save_token='_diffInit'
fi
test_bt=$feature_dim
ckpt_path='./results/'$ckpt_path
fig_dir='./figs/dim'$feature_dim'_lmbda'$lambda'_bt'$bt$save_token
fSinVals=$fig_dir'/sinVals'
save_feats=1
fsave_feats='./saved_feats/dim'$feature_dim'_lmbda'$lambda'_bt'$bt$save_token

WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=1 \
  python main.py \
    --test-only=1 \
    --dataset=$dataset \
    --feature_dim=$feature_dim \
    --proj-head-type=$proj_head_type \
    --batch_size=$test_bt \
    --load-ckpt=0 \
    --pretrained-path=$ckpt_path \
    --fSinVals=$fSinVals \
    --save-feats=$save_feats \
    --fsave-feats=$fsave_feats

