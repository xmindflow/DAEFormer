#!/bin/bash
cd /work/scratch/azad/TransFiLM/
chmod +x train.sh
/work/scratch/azad/anaconda3/envs/pytorch_cuda11/bin/python /work/scratch/azad/TransFiLM/train.py "$@"