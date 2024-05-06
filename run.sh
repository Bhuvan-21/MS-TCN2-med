#!/bin/bash
python main.py --action=train --dataset=${1} --split=${2} --num_epochs=${3} \
               --num_layers_PG=${9-11} \
               --num_layers_R=${10-10} \
               --num_R=${11-3} \
               --features_dim=${4} \
               --loss_mse=${5:-0.15} \
               --loss_dice=${6:-0.0} \
               --loss_focal=${7:-0.0} \
               --weights=${8-None}

python main.py --action=predict --dataset=${1} --split=${2} --num_epochs=${3} \
               --num_layers_PG=${9-11} \
               --num_layers_R=${10-10} \
               --num_R=${11-3} \
               --features_dim=${4} \
               --loss_mse=${5:-0.15} \
               --loss_dice=${6:-0.0} \
               --loss_focal=${7:-0.0} \
               --weights=${8-None}

python eval.py --dataset=${1} --split=${2} 
