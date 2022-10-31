#!/bin/bash
cd /home/students/arimond/SwinUnet/
chmod +x train.sh
/home/students/arimond/miniconda3/envs/new_cuda/bin/python /home/students/arimond/EffFormer/train_eff.py  "$@"