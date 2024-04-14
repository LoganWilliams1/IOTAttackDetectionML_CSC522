#!/bin/bash
#BSUB -n 16
#BSUB -W 15:00
#BSUB -J IoT_regression
#BSUB -o regression_outputs/stdout.%J
#BSUB -e regression_outputs/stderr.%J

source ~/.bashrc
conda activate /usr/local/usrapps/csc522s24/lrwilli7/env_synth_2
python ../regression/regression_synthetic.py
conda deactivate