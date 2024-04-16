#!/bin/bash
#BSUB -n 16
#BSUB -W 24:00
#BSUB -J IoT_Viz
#BSUB -o dnn_outputs/stdout.%J
#BSUB -e dnn_outputs/stderr.%J

source ~/.bashrc
conda activate /usr/local/usrapps/csc522s24/lrwilli7/env_synth_2
python ../data_breakdown/sdv_visualization.py
conda deactivate