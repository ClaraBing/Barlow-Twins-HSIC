#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

token=''

case=7
feature_dim=128
fname_linear_cls=''
if [ $case -eq 1 ]; then
  lmbda=0.005
  model_path='0.005_128_128_cifar10_model.pth'
  token='_lmbda'$lmbda
elif [ $case -eq 2 ]; then
  lmbda=0.05
  model_path='lmbda0.05_dim128/0.05_128_128_cifar10_model.pth'
  token='_lmbda'$lmbda
elif [ $case -eq 3 ]; then
  feature_dim=256
  lmbda=0.05
  # last ckpt
  model_path='0.05_256_128_cifar10_model.pth'
  token='_lmbda'$lmbda'_eLast'
  # ckpt in the middle
  # model_path='0.05_256_128_cifar10_model_150.pth'
  # token='_e150'

  model_subdir='lmbda0.05_dim256'
  model_path=$model_subdir'/'$model_path
elif [ $case -eq 4 ]; then
  lmbda=0.005
  model_path='cifar10_linear_feat128_lmbda0.005_lr1e-3_wd1e-6_bt128_sameInit2/model.pth'
  token='_lmbda'$lmbda'_sameInit2'
elif [ $case -eq 5 ]; then
  lmbda=0.005
  model_path='cifar10_linear_feat128_lmbda0.005_lr1e-3_wd1e-6_bt128_sameInit2/model.pth'
  fname_linear_cls='./results_linear/cifar10_linear_linear_feat128_lmbda0.05_lr3e-4_wd1e-5_bt128_linear_model.pth'
  token='_lmbda0.005with0.05'
elif [ $case -eq 6 ]; then
  lmbda=0.05
  model_path='lmbda0.05_dim128/0.05_128_128_cifar10_model.pth'
  fname_linear_cls='./results_linear/cifar10_linear_linear_feat128_lmbda0.005_lr1e-4_wd1e-5_bt128_sameInit2_linear_model.pth'
  token='_lmbda0.05with0.005'
elif [ $case -eq 7 ]; then
  # BYOL backbone, lmbda0.05 linear
  model_path='byol_128_512_512_cifar10_model_1000_Ashwini.pth'
  fname_linear_cls='./results_linear/cifar10_linear_linear_feat128_lmbda0.05_lr3e-4_wd1e-5_bt128_linear_model.pth'
  token='_BYOLwith0.05_run2_tmp'
  wb_token='byol'
fi
model_path='./results/'$model_path
# token=$token'_tmp'

for run in 1 2
do
for bt in 128
do
for wd in 1e-5
do
for lr in 1e-3
# 1e-2 3e-4
do
wb_name=$dataset'_linear_'$proj_head_type'_feat'$feature_dim'_lr'$lr'_wd'$wd'_bt'$bt$token
WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=1 \
  python linear.py \
    --dataset=$dataset \
    --lr=$lr \
    --wd=$wd \
    --batch_size=$bt \
    --model_path=$model_path \
    --fname-linear-cls=$fname_linear_cls \
    --wb-name=$wb_name \
    --wb-token=$wb_token
done
done
done
done
