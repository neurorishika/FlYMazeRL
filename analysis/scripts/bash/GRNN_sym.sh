#!/bin/bash

bsub -n1 -J "GRNNx2-0-1" -o "GRNNx2-0-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx2-1-1" -o "GRNNx2-1-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx2-2-1" -o "GRNNx2-2-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx2-3-1" -o "GRNNx2-3-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx2-4-1" -o "GRNNx2-4-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx2-5-1" -o "GRNNx2-5-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx2-6-1" -o "GRNNx2-6-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx2-7-1" -o "GRNNx2-7-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx2-8-1" -o "GRNNx2-8-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx2-9-1" -o "GRNNx2-9-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx3-0-1" -o "GRNNx3-0-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx3-1-1" -o "GRNNx3-1-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx3-2-1" -o "GRNNx3-2-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx3-3-1" -o "GRNNx3-3-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx3-4-1" -o "GRNNx3-4-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx3-5-1" -o "GRNNx3-5-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx3-6-1" -o "GRNNx3-6-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx3-7-1" -o "GRNNx3-7-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx3-8-1" -o "GRNNx3-8-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx3-9-1" -o "GRNNx3-9-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 3 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx5-0-1" -o "GRNNx5-0-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx5-1-1" -o "GRNNx5-1-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx5-2-1" -o "GRNNx5-2-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx5-3-1" -o "GRNNx5-3-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx5-4-1" -o "GRNNx5-4-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx5-5-1" -o "GRNNx5-5-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx5-6-1" -o "GRNNx5-6-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx5-7-1" -o "GRNNx5-7-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx5-8-1" -o "GRNNx5-8-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx5-9-1" -o "GRNNx5-9-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx10-0-1" -o "GRNNx10-0-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx10-1-1" -o "GRNNx10-1-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx10-2-1" -o "GRNNx10-2-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx10-3-1" -o "GRNNx10-3-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx10-4-1" -o "GRNNx10-4-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx10-5-1" -o "GRNNx10-5-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx10-6-1" -o "GRNNx10-6-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx10-7-1" -o "GRNNx10-7-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx10-8-1" -o "GRNNx10-8-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx10-9-1" -o "GRNNx10-9-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx100-0-1" -o "GRNNx100-0-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx100-1-1" -o "GRNNx100-1-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx100-2-1" -o "GRNNx100-2-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx100-3-1" -o "GRNNx100-3-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx100-4-1" -o "GRNNx100-4-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx100-5-1" -o "GRNNx100-5-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx100-6-1" -o "GRNNx100-6-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx100-7-1" -o "GRNNx100-7-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx100-8-1" -o "GRNNx100-8-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx100-9-1" -o "GRNNx100-9-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx200-0-1" -o "GRNNx200-0-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx200-1-1" -o "GRNNx200-1-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx200-2-1" -o "GRNNx200-2-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx200-3-1" -o "GRNNx200-3-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx200-4-1" -o "GRNNx200-4-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx200-5-1" -o "GRNNx200-5-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx200-6-1" -o "GRNNx200-6-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx200-7-1" -o "GRNNx200-7-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx200-8-1" -o "GRNNx200-8-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
bsub -n1 -J "GRNNx200-9-1" -o "GRNNx200-9-1.output" "python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size 200 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric yes"
sleep .5
