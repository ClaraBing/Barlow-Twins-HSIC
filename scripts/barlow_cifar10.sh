#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

token=''

# whether to force the diag entries to be -1 (rather than 1)
corr_neg_one_on_diag=0
if [ $corr_neg_one_on_diag = 1 ]; then
  token=$token'diagNeg'
fi
loss_no_on_diag=0
if [ $loss_no_on_diag = 1 ]; then
  token=$token'_noOnDiag'
fi
loss_no_off_diag=0
if [ $loss_no_off_diag = 1 ]; then
  token=$token'_noOffDiag'
fi
norm_std=1
if [ $norm_std = 0 ]; then
  token=$token'_noNormStd'
fi

load_ckpt=0
model_dir='results/'
pretrained_path='cifar10_linear_feat128_lmbda0.005_lr1e-3_wd1e-6_bt128_model_init.pth'
pretrained_path=$model_dir$pretrained_path
token='_sameInit'
token='_tmp'

bt=128

for lr in 1e-3
do
for wd in 1e-6
do
for feature_dim in 128
do
for lambda in 0.005
do

wb_name=$dataset'_'$proj_head_type'_feat'$feature_dim'_lmbda'$lambda'_lr'$lr'_wd'$wd'_bt'$bt$token

WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=0 \
  python main.py \
    --dataset=$dataset \
    --corr_neg_one_on_diag=$corr_neg_one_on_diag \
    --feature_dim=$feature_dim \
    --proj-head-type=$proj_head_type \
    --lr=$lr \
    --wd=$wd \
    --batch_size=$bt \
    --loss-no-on-diag=$loss_no_on_diag \
    --loss-no-off-diag=$loss_no_off_diag \
    --norm-std=$norm_std \
    --lmbda=$lambda \
    --load-ckpt=$load_ckpt \
    --pretrained-path=$pretrained_path \
    --wb-name=$wb_name
done
done
done
done
