#!/bin/bash

bsub -n1 -J "GRNNx2-0-0" -o "GRNNx2-0-0.output" "python ../nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GRNNx3-0-0" -o "GRNNx3-0-0.output" "python ../nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GRNNx5-0-0" -o "GRNNx5-0-0.output" "python ../nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GRNNx10-0-0" -o "GRNNx10-0-0.output" "python ../nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GRNNx100-0-0" -o "GRNNx100-0-0.output" "python ../nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
bsub -n1 -J "GRNNx200-0-0" -o "GRNNx200-0-0.output" "python ../nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric no --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohantas2022/ "
sleep .5
