#!/bin/bash

dataset='cifar10'
# proj_head_type='linear'
proj_head_type='linear_noBNReLU'

case=3
feature_dim=512
proj_hidden_dim=512
fname_linear_cls=''
if [ $case -eq 1 ]; then
  model_path='byol_cifar10_linear_feat512_temp0.5_lr1e-3_wd1e-6_bt512_model.pth'
  token=''

  model_subdir='byol_cifar10_linear_feat512_temp0.5_lr1e-3_wd1e-6_bt512'
  model_path=$model_subdir'/'$model_path
elif [ $case -eq 2 ]; then
  token='_randInit'
  model_path=''
elif [ $case -eq 3 ]; then
  # Ashwini's checkpoint
  model_path='byol_128_512_512_cifar10_model_1000_Ashwini.pth'
  proj_head_type='linear_noBNReLU'
  proj_hidden_dim=128
  token='_run3'
elif [ $case -eq 4 ]; then
  # BYOL backbone, lmbda0.05 linear
  model_path='byol_128_512_512_cifar10_model_1000_Ashwini.pth'
  fname_linear_cls='./results_linear/cifar10_linear_linear_feat128_lmbda0.05_lr3e-4_wd1e-5_bt128_linear_model.pth'
  token='_BYOLwith0.05'
  wb_token='byol'
elif [ $case -eq 5 ]; then
  # BYOL backbone, lmbda0.005 linear
  model_path='byol_128_512_512_cifar10_model_1000_Ashwini.pth'
  fname_linear_cls='./results_linear/cifar10_linear_linear_feat128_lmbda0.005_lr1e-4_wd1e-5_bt128_sameInit2_linear_model.pth'
  token='_BYOLwith0.005'
  wb_token='byol'
elif [ $case -eq 6 ]; then
  # lmdba=0.05 backbone, BYOL linear (lr=1e-2)
  model_path='lmbda0.05_dim128/0.05_128_128_cifar10_model.pth'
  fname_linear_cls='./results_linear/byol_cifar10_linear_linear_noBNReLU_feat512_lr1e-2_wd1e-5_bt128_linear_model.pth'
  token='_lmbda0.05withBYOL'
  wb_token='byol'
elif [ $case -eq 7 ]; then
  # lmdba=0.05 backbone, BYOL linear (lr=1e-3)
  model_path='lmbda0.05_dim128/0.05_128_128_cifar10_model.pth'
  fname_linear_cls='./results_linear/byol_cifar10_linear_linear_noBNReLU_feat512_lr1e-3_wd1e-5_bt128_linear_model.pth'
  token='_lmbda0.05withBYOLlr1e-3'
  wb_token='byol'
elif [ $case -eq 8 ]; then
  # lmdba=0.005 backbone, BYOL linear (lr=1e-2)
  model_path='cifar10_linear_feat128_lmbda0.005_lr1e-3_wd1e-6_bt128_sameInit2/model.pth'
  fname_linear_cls='./results_linear/byol_cifar10_linear_linear_noBNReLU_feat512_lr1e-2_wd1e-5_bt128_linear_model.pth'
  token='_lmbda0.005withBYOL'
  wb_token='byol'
fi
model_path='./results/'$model_path

epochs=100

for bt in 128
do
for wd in 1e-5
do
for lr in 1e-2
# 1e-3 1e-1
do
wb_name='byol_'$dataset'_linear_'$proj_head_type'_feat'$feature_dim'_lr'$lr'_wd'$wd'_bt'$bt$token
# WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=0 \
  python byol_linear.py \
    --dataset=$dataset \
    --feature_dim=$feature_dim \
    --proj_hidden_dim=$proj_hidden_dim \
    --proj-head-type=$proj_head_type \
    --lr=$lr \
    --wd=$wd \
    --epochs=$epochs \
    --batch_size=$bt \
    --model_path=$model_path \
    --fname-linear-cls=$fname_linear_cls \
    --wb-name=$wb_name \
    --wb-token=$wb_token
# exit # TODO: this is for debugging
done
done
done
