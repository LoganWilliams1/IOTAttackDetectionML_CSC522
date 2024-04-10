#!/bin/bash
#BSUB -n 16
#BSUB -W 48:00
#BSUB -J IoT_DNN
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -o dnn_outputs/stdout.%J
#BSUB -e dnn_outputs/stderr.%J

source ~/.bashrc
conda activate /usr/local/usrapps/csc522s24/lrwilli7/env_synth
python ../neuralnetwork/dnn_synthetic.py
conda deactivate