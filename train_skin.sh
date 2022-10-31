#!/bin/bash
cd /home/students/arimond/EffFormer/
chmod +x train_skin.sh
/home/students/arimond/miniconda3/envs/new_cuda/bin/python /home/students/arimond/EffFormer/train_test_skin.py  "$@"