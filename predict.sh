python main.py --action=predict --dataset=${1} --split=${2} --num_epochs=${3} \
               --num_layers_PG=${10-13} \
               --num_layers_R=${11-12} \
               --num_R=${12-4} \
               --features_dim=${4:-1024} \
               --loss_mse=${5:-0.15} \
               --loss_dice=${6:-0.0} \
               --loss_focal=${7:-0.0} \
               --weights=${8-None} \
               --weights_coeff=${9-1.0} \
               --device=cuda \
               #--adaptive_mse