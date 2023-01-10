#!/bin/bash

bsub -n1 -J "GRNNx2-0-1" -o "GRNNx2-0-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx2-1-1" -o "GRNNx2-1-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx2-2-1" -o "GRNNx2-2-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx2-3-1" -o "GRNNx2-3-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx3-0-1" -o "GRNNx3-0-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx3-1-1" -o "GRNNx3-1-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx3-2-1" -o "GRNNx3-2-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx3-3-1" -o "GRNNx3-3-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx5-0-1" -o "GRNNx5-0-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx5-1-1" -o "GRNNx5-1-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx5-2-1" -o "GRNNx5-2-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx5-3-1" -o "GRNNx5-3-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx10-0-1" -o "GRNNx10-0-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx10-1-1" -o "GRNNx10-1-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx10-2-1" -o "GRNNx10-2-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx10-3-1" -o "GRNNx10-3-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx100-0-1" -o "GRNNx100-0-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx100-1-1" -o "GRNNx100-1-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx100-2-1" -o "GRNNx100-2-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx100-3-1" -o "GRNNx100-3-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx200-0-1" -o "GRNNx200-0-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx200-1-1" -o "GRNNx200-1-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx200-2-1" -o "GRNNx200-2-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
bsub -n1 -J "GRNNx200-3-1" -o "GRNNx200-3-1.output" "python ../nn_fitting_mohanta.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 1 --n_ensemble 5 --early_stopping 2500 --symmetric yes --save_path /groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/nn/mohanta2022/ "
sleep .5
