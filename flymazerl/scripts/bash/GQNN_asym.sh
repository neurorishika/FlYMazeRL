#!/bin/bash

bsub -n1 -J "GQNNx2-0-0" -o "GQNNx2-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2-1-0" -o "GQNNx2-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2-2-0" -o "GQNNx2-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2-3-0" -o "GQNNx2-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2-4-0" -o "GQNNx2-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2-5-0" -o "GQNNx2-5-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2-6-0" -o "GQNNx2-6-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2-7-0" -o "GQNNx2-7-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2-8-0" -o "GQNNx2-8-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2-9-0" -o "GQNNx2-9-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5-0-0" -o "GQNNx5-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5-1-0" -o "GQNNx5-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5-2-0" -o "GQNNx5-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5-3-0" -o "GQNNx5-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5-4-0" -o "GQNNx5-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5-5-0" -o "GQNNx5-5-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5-6-0" -o "GQNNx5-6-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5-7-0" -o "GQNNx5-7-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5-8-0" -o "GQNNx5-8-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5-9-0" -o "GQNNx5-9-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10-0-0" -o "GQNNx10-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10-1-0" -o "GQNNx10-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10-2-0" -o "GQNNx10-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10-3-0" -o "GQNNx10-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10-4-0" -o "GQNNx10-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10-5-0" -o "GQNNx10-5-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10-6-0" -o "GQNNx10-6-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10-7-0" -o "GQNNx10-7-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10-8-0" -o "GQNNx10-8-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10-9-0" -o "GQNNx10-9-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2x2-0-0" -o "GQNNx2x2-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2x2-1-0" -o "GQNNx2x2-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2x2-2-0" -o "GQNNx2x2-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2x2-3-0" -o "GQNNx2x2-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2x2-4-0" -o "GQNNx2x2-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2x2-5-0" -o "GQNNx2x2-5-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2x2-6-0" -o "GQNNx2x2-6-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2x2-7-0" -o "GQNNx2x2-7-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2x2-8-0" -o "GQNNx2x2-8-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx2x2-9-0" -o "GQNNx2x2-9-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5x5-0-0" -o "GQNNx5x5-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5x5-1-0" -o "GQNNx5x5-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5x5-2-0" -o "GQNNx5x5-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5x5-3-0" -o "GQNNx5x5-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5x5-4-0" -o "GQNNx5x5-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5x5-5-0" -o "GQNNx5x5-5-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5x5-6-0" -o "GQNNx5x5-6-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5x5-7-0" -o "GQNNx5x5-7-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5x5-8-0" -o "GQNNx5x5-8-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx5x5-9-0" -o "GQNNx5x5-9-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10-0-0" -o "GQNNx10x10-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10-1-0" -o "GQNNx10x10-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10-2-0" -o "GQNNx10x10-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10-3-0" -o "GQNNx10x10-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10-4-0" -o "GQNNx10x10-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10-5-0" -o "GQNNx10x10-5-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10-6-0" -o "GQNNx10x10-6-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10-7-0" -o "GQNNx10x10-7-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10-8-0" -o "GQNNx10x10-8-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10-9-0" -o "GQNNx10x10-9-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100-0-0" -o "GQNNx100x100-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100-1-0" -o "GQNNx100x100-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100-2-0" -o "GQNNx100x100-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100-3-0" -o "GQNNx100x100-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100-4-0" -o "GQNNx100x100-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100-5-0" -o "GQNNx100x100-5-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100-6-0" -o "GQNNx100x100-6-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100-7-0" -o "GQNNx100x100-7-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100-8-0" -o "GQNNx100x100-8-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100-9-0" -o "GQNNx100x100-9-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10x10-0-0" -o "GQNNx10x10x10-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10x10-1-0" -o "GQNNx10x10x10-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10x10-2-0" -o "GQNNx10x10x10-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10x10-3-0" -o "GQNNx10x10x10-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10x10-4-0" -o "GQNNx10x10x10-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10x10-5-0" -o "GQNNx10x10x10-5-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10x10-6-0" -o "GQNNx10x10x10-6-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10x10-7-0" -o "GQNNx10x10x10-7-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10x10-8-0" -o "GQNNx10x10x10-8-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx10x10x10-9-0" -o "GQNNx10x10x10-9-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100x100-0-0" -o "GQNNx100x100x100-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100x100-1-0" -o "GQNNx100x100x100-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100x100-2-0" -o "GQNNx100x100x100-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100x100-3-0" -o "GQNNx100x100x100-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100x100-4-0" -o "GQNNx100x100x100-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100x100-5-0" -o "GQNNx100x100x100-5-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100x100-6-0" -o "GQNNx100x100x100-6-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100x100-7-0" -o "GQNNx100x100x100-7-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100x100-8-0" -o "GQNNx100x100x100-8-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
bsub -n1 -J "GQNNx100x100x100-9-0" -o "GQNNx100x100x100-9-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric no"
sleep .5
