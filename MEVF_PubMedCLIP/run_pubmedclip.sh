#!/bin/bash


#####******RAD

python main.py --model BAN --use_RAD --RAD_dir data_RAD --clip --clip_path "path/to/fine-tuned/PubMedCLIP" --autoencoder --output saved_models/BAN_CLIPViT_32batchsize_lr002_nondeterministic --lr 0.002 --batch_size 32 --feat_dim_clip 576 2>&1 | tee logs/RAD_BAN_CLIPViT_32batchsize_lr002_nondeterministic.out

python main.py --model BAN --use_RAD --RAD_dir data_RAD --clip --autoencoder --output saved_models/BAN_CLIPRN50_32batchsize_nondeterministic --result_output results/results_RAD_BAN_CLIPRN50_32batchsize__nondeterministic --lr 0.002 --batch_size 32 --clip_vision_encoder "RN50" --clip_path "path/to/fine-tuned/PubMedCLIP" --feat_dim_clip 1088 2>&1 | tee logs/RAD_BAN_CLIPRN50_32batchsize_lr002_nondeterministic.out

python main.py --model BAN --use_RAD --RAD_dir data_RAD --clip --autoencoder --output saved_models/BAN_CLIPRN50x4_16batchsize_lr002_nondeterministic --result_output results/results_RAD_BAN_CLIPRN50x4_16batchsize_lr002_nondeterministic --lr 0.002 --batch_size 16 --clip_vision_encoder "RN50x4" --clip_path "path/to/fine-tuned/PubMedCLIP" --feat_dim_clip 704 2>&1 | tee logs/RAD_BAN_CLIPRN50x4_16batchsize_lr002_nondeterministic.out


#####******SLAKE

python main.py --model BAN --use_SLAKE --SLAKE_dir data_SLAKE --clip --clip_path "path/to/fine-tuned/PubMedCLIP" --autoencoder --output saved_models/SLAKE_BAN_CLIPViT_32batchsize_lr002_nondeterministic --lr 0.002 --batch_size 32 --feat_dim_clip 576 2>&1 | tee logs/SLAKE_BAN_CLIPViT_32batchsize_lr002_nondeterministic.out

python main.py --model BAN --use_SLAKE --SLAKE_dir data_SLAKE --clip --autoencoder --output saved_models/SLAKE_BAN_CLIPRN50_32batchsize_lr002_nondeterministic --result_output results/results_SLAKE_BAN_CLIPRN50_32batchsize_lr002_nondeterministic --lr 0.002 --batch_size 32 --clip_vision_encoder "RN50" --clip_path "path/to/fine-tuned/PubMedCLIP" --feat_dim_clip 1088 2>&1 | tee logs/SLAKE_BAN_CLIPRN50_32batchsize_lr002_nondeterministic.out

python main.py --model BAN --use_SLAKE --SLAKE_dir data_SLAKE --clip --autoencoder --output saved_models/SLAKE_BAN_CLIPRN50x4_16batchsize_lr002_nondeterministic --result_output results/results_SLAKE_BAN_CLIPRN50x4_16batchsize_lr002_nondeterministic --lr 0.002 --batch_size 16 --clip_vision_encoder "RN50x4" --clip_path "path/to/fine-tuned/PubMedCLIP" --feat_dim_clip 704 2>&1 | tee logs/SLAKE_BAN_CLIPRN50x4_16batchsize_lr002_nondeterministic.out
