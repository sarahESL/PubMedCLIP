#!/bin/bash


####*****RAD

python main.py --model BAN --use_RAD --RAD_dir data_RAD --clip --autoencoder --clip_org --output saved_models/BAN_CLIPOrgViT_32batchsize_lr002_nondeterministic --lr 0.002 --batch_size 32 --feat_dim_clip 576 2>&1 | tee logs/RAD_BAN_CLIPOrgViT_32batchsize_lr002_nondeterministic.out

python main.py --model BAN --use_RAD --RAD_dir data_RAD --clip --clip_org --autoencoder --output saved_models/BAN_CLIPOrgRN50_32batchsize_lr002_nondeterministic --result_output results/results_RAD_BAN_CLIPOrgRN50_32batchsize_lr002_nondeterministic --lr 0.002 --batch_size 32 --clip_vision_encoder "RN50" --feat_dim_clip 1088 2>&1 | tee logs/RAD_BAN_CLIPOrgRN50_32batchsize_lr002_nondeterministic.out

python main.py --model BAN --use_RAD --RAD_dir data_RAD --clip --clip_org --autoencoder --output saved_models/BAN_CLIPOrgRN50x4_32batchsize_lr002_nondeterministic --result_output results/results_RAD_BAN_CLIPOrgRN50x4_32batchsize_lr002_nondeterministic --lr 0.002 --batch_size 32 --clip_vision_encoder "RN50x4" --feat_dim_clip 704 2>&1 | tee logs/RAD_BAN_CLIPOrgRN50x4_32batchsize_lr002_nondeterministic.out


####*****SLAKE

python main.py --model BAN --use_SLAKE --SLAKE_dir data_SLAKE --clip --clip_org --autoencoder --output saved_models/SLAKE_BAN_CLIPOrgViT_32batchsize_lr002_nondeterministic --lr 0.002 --batch_size 32 --feat_dim_clip 576 2>&1 | tee logs/SLAKE_BAN_CLIPOrgViT_32batchsize_lr002_nondeterministic.out

python main.py --model BAN --use_SLAKE --SLAKE_dir data_SLAKE --clip --clip_org --autoencoder --output saved_models/SLAKE_BAN_CLIPOrgRN50_32batchsize_lr002_nondeterministic --result_output results/results_SLAKE_BAN_CLIPOrgRN50_32batchsize_lr002_nondeterministic --lr 0.002 --batch_size 32 --clip_vision_encoder "RN50" --feat_dim_clip 1088 2>&1 | tee logs/SLAKE_BAN_CLIPOrgRN50_32batchsize_lr002_nondeterministic.out

python main.py --model BAN --use_SLAKE --SLAKE_dir data_SLAKE --clip --clip_org --autoencoder --output saved_models/SLAKE_BAN_CLIPOrgRN50x4_32batchsize_lr002_nondeterministic --result_output results/results_SLAKE_BAN_CLIPOrgRN50x4_32batchsize_lr002_nondeterministic --lr 0.002 --batch_size 32 --clip_vision_encoder "RN50x4" --feat_dim_clip 704 2>&1 | tee logs/SLAKE_BAN_CLIPOrgRN50x4_32batchsize_lr002_nondeterministic.out
