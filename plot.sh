#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../../../..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

python $PROJECT_DIR/src/plot_avg.py --exp_name ppo_mlp_v2_overparam320_metaworld_10_tasks --algos agem_ref_grad_batch_size5000

python $PROJECT_DIR/src/plot_continual.py --exp_name ppo_mlp_v2_overparam320_metaworld_10_tasks --algos agem_ref_grad_batch_size5000

python $PROJECT_DIR/src/plot_avg.py --exp_name ppo_mlp_v2_overparam320_metaworld_10_tasks --algos agem_ref_grad_batch_size5000_noobs

python plot_avg.py --exp_name pearlcontinual --algos 2021_08_13_11_27_06


