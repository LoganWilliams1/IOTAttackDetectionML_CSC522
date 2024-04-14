#!/bin/bash
#BSUB -n 32
#BSUB -W 48:00
#BSUB -J IoT_regression
#BSUB -R "rusage[mem=128]"
#BSUB -o regression_outputs/stdout.%J
#BSUB -e regression_outputs/stderr.%J

source ~/.bashrc
conda activate /usr/local/usrapps/csc522s24/lrwilli7/env_synth
python ../synthetic_data/GCsynth_regression.py
conda deactivate