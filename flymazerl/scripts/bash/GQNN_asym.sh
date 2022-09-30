#!/bin/bash

bsub -n1 -J "GQNNx2-0-0" -o "GQNNx2-0-0.output" "python ../nn_fitting_rajagopalan.py --agent GQNN --hidden_state_sizes 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GQNNx5-0-0" -o "GQNNx5-0-0.output" "python ../nn_fitting_rajagopalan.py --agent GQNN --hidden_state_sizes 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GQNNx10-0-0" -o "GQNNx10-0-0.output" "python ../nn_fitting_rajagopalan.py --agent GQNN --hidden_state_sizes 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GQNNx2x2-0-0" -o "GQNNx2x2-0-0.output" "python ../nn_fitting_rajagopalan.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GQNNx5x5-0-0" -o "GQNNx5x5-0-0.output" "python ../nn_fitting_rajagopalan.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GQNNx10x10-0-0" -o "GQNNx10x10-0-0.output" "python ../nn_fitting_rajagopalan.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GQNNx100x100-0-0" -o "GQNNx100x100-0-0.output" "python ../nn_fitting_rajagopalan.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GQNNx10x10x10-0-0" -o "GQNNx10x10x10-0-0.output" "python ../nn_fitting_rajagopalan.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GQNNx100x100x100-0-0" -o "GQNNx100x100x100-0-0.output" "python ../nn_fitting_rajagopalan.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
