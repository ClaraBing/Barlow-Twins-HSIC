#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

token=''

# TODO: do we want to add an option for not using std?
norm_std=1
if [ $norm_std = 0 ]; then
  token=$token'_noNormStd'
fi

load_ckpt=0
model_dir='results/'
pretrained_path=''
pretrained_path=$model_dir$pretrained_path
token=$token

bt=128
proj_hidden_sim=128
temperature=0.5

for lr in 1e-3
do
for wd in 1e-6
do
for feature_dim in 128
do
for lambda in 0.05
do

wb_name='byol_'$dataset'_'$proj_head_type'_feat'$feature_dim'_temp'$temperature'_lr'$lr'_wd'$wd'_bt'$bt$token

WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=0 \
  python byol_main.py \
    --dataset=$dataset \
    --feature_dim=$feature_dim \
    --proj-head-type=$proj_head_type \
    --lr=$lr \
    --wd=$wd \
    --temperature=$temperature \
    --batch_size=$bt \
    --load-ckpt=$load_ckpt \
    --pretrained-path=$pretrained_path \
    --wb-name=$wb_name
done
done
done
done
