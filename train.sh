#!/bin/bash

python main.py --action=train --dataset=${1} --split=${2} \
                --num_epochs=100 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \ 
                --features_dim=${3} \
                --loss_lambda=${4:-0.15} \
                --loss_dice=${5:-0.0}
