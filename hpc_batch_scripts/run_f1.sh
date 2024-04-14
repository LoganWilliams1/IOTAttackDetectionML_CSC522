#!/bin/bash
#BSUB -n 16
#BSUB -W 15:00
#BSUB -J IoT_DNN_f1
#BSUB -o dnn_outputs/stdout.%J
#BSUB -e dnn_outputs/stderr.%J

source ~/.bashrc
conda activate /usr/local/usrapps/csc522s24/lrwilli7/env_synth_2
python ../neuralnetwork/f1_score.py
conda deactivate