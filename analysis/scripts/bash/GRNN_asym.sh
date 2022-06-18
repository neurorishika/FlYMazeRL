#!/bin/bash

bsub -n1 -J "GRNNx2-0-0" -o "GRNNx2-0-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx2-1-0" -o "GRNNx2-1-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx2-2-0" -o "GRNNx2-2-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx2-3-0" -o "GRNNx2-3-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx2-4-0" -o "GRNNx2-4-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx3-0-0" -o "GRNNx3-0-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx3-1-0" -o "GRNNx3-1-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx3-2-0" -o "GRNNx3-2-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx3-3-0" -o "GRNNx3-3-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx3-4-0" -o "GRNNx3-4-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx5-0-0" -o "GRNNx5-0-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx5-1-0" -o "GRNNx5-1-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx5-2-0" -o "GRNNx5-2-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx5-3-0" -o "GRNNx5-3-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx5-4-0" -o "GRNNx5-4-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx10-0-0" -o "GRNNx10-0-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx10-1-0" -o "GRNNx10-1-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx10-2-0" -o "GRNNx10-2-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx10-3-0" -o "GRNNx10-3-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx10-4-0" -o "GRNNx10-4-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx100-0-0" -o "GRNNx100-0-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx100-1-0" -o "GRNNx100-1-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx100-2-0" -o "GRNNx100-2-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx100-3-0" -o "GRNNx100-3-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx100-4-0" -o "GRNNx100-4-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx200-0-0" -o "GRNNx200-0-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx200-1-0" -o "GRNNx200-1-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx200-2-0" -o "GRNNx200-2-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx200-3-0" -o "GRNNx200-3-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GRNNx200-4-0" -o "GRNNx200-4-0.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
