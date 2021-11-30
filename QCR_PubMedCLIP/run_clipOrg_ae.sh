#!/bin/bash

####*****RAD

python main/main.py --cfg configs/qcr_clipOrgRN50_ae_rad_16batchsize_withtfidf_nondeterministic.yaml
python main/main.py --cfg configs/qcr_clipOrgRN50x4_ae_rad_16batchsize_withtfidf_nondeterministic.yaml
python main/main.py --cfg configs/qcr_clipOrgViT_ae_rad_16batchsize_withtfidf_nondeterministic.yaml


####*****SLAKE

python main/main.py --cfg configs/qcr_clipOrgRN50_ae_slake_16batchsize_withtfidf_nondeterministic.yaml
python main/main.py --cfg configs/qcr_clipOrgRN50x4_ae_slake_16batchsize_withtfidf_nondeterministic.yaml
python main/main.py --cfg configs/qcr_clipOrgViT_ae_slake_16batchsize_withtfidf_nondeterministic.yaml
