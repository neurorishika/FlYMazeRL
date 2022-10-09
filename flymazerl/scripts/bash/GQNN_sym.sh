#!/bin/bash

bsub -n1 -J "GQNNx2-0-1" -o "GQNNx2-0-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx2-1-1" -o "GQNNx2-1-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx2-2-1" -o "GQNNx2-2-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx2-3-1" -o "GQNNx2-3-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx5-0-1" -o "GQNNx5-0-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx5-1-1" -o "GQNNx5-1-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx5-2-1" -o "GQNNx5-2-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx5-3-1" -o "GQNNx5-3-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx10-0-1" -o "GQNNx10-0-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx10-1-1" -o "GQNNx10-1-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx10-2-1" -o "GQNNx10-2-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx10-3-1" -o "GQNNx10-3-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx2x2-0-1" -o "GQNNx2x2-0-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx2x2-1-1" -o "GQNNx2x2-1-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx2x2-2-1" -o "GQNNx2x2-2-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx2x2-3-1" -o "GQNNx2x2-3-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx5x5-0-1" -o "GQNNx5x5-0-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx5x5-1-1" -o "GQNNx5x5-1-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx5x5-2-1" -o "GQNNx5x5-2-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx5x5-3-1" -o "GQNNx5x5-3-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx10x10-0-1" -o "GQNNx10x10-0-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx10x10-1-1" -o "GQNNx10x10-1-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx10x10-2-1" -o "GQNNx10x10-2-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx10x10-3-1" -o "GQNNx10x10-3-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx100x100-0-1" -o "GQNNx100x100-0-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx100x100-1-1" -o "GQNNx100x100-1-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx100x100-2-1" -o "GQNNx100x100-2-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx100x100-3-1" -o "GQNNx100x100-3-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx10x10x10-0-1" -o "GQNNx10x10x10-0-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx10x10x10-1-1" -o "GQNNx10x10x10-1-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx10x10x10-2-1" -o "GQNNx10x10x10-2-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx10x10x10-3-1" -o "GQNNx10x10x10-3-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx100x100x100-0-1" -o "GQNNx100x100x100-0-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx100x100x100-1-1" -o "GQNNx100x100x100-1-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx100x100x100-2-1" -o "GQNNx100x100x100-2-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GQNNx100x100x100-3-1" -o "GQNNx100x100x100-3-1.output" "python ../nn_fitting_mohanta.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
