#!/bin/bash

bsub -n1 -J "GQNNx2-0-0" -o "GQNNx2-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx2-1-0" -o "GQNNx2-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx2-2-0" -o "GQNNx2-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx2-3-0" -o "GQNNx2-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx2-4-0" -o "GQNNx2-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx5-0-0" -o "GQNNx5-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx5-1-0" -o "GQNNx5-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx5-2-0" -o "GQNNx5-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx5-3-0" -o "GQNNx5-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx5-4-0" -o "GQNNx5-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10-0-0" -o "GQNNx10-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10-1-0" -o "GQNNx10-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10-2-0" -o "GQNNx10-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10-3-0" -o "GQNNx10-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10-4-0" -o "GQNNx10-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx2x2-0-0" -o "GQNNx2x2-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx2x2-1-0" -o "GQNNx2x2-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx2x2-2-0" -o "GQNNx2x2-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx2x2-3-0" -o "GQNNx2x2-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx2x2-4-0" -o "GQNNx2x2-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 2 2 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx5x5-0-0" -o "GQNNx5x5-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx5x5-1-0" -o "GQNNx5x5-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx5x5-2-0" -o "GQNNx5x5-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx5x5-3-0" -o "GQNNx5x5-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx5x5-4-0" -o "GQNNx5x5-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 5 5 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10x10-0-0" -o "GQNNx10x10-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10x10-1-0" -o "GQNNx10x10-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10x10-2-0" -o "GQNNx10x10-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10x10-3-0" -o "GQNNx10x10-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10x10-4-0" -o "GQNNx10x10-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx100x100-0-0" -o "GQNNx100x100-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx100x100-1-0" -o "GQNNx100x100-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx100x100-2-0" -o "GQNNx100x100-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx100x100-3-0" -o "GQNNx100x100-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx100x100-4-0" -o "GQNNx100x100-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10x10x10-0-0" -o "GQNNx10x10x10-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10x10x10-1-0" -o "GQNNx10x10x10-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10x10x10-2-0" -o "GQNNx10x10x10-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10x10x10-3-0" -o "GQNNx10x10x10-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx10x10x10-4-0" -o "GQNNx10x10x10-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 10 10 10 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx100x100x100-0-0" -o "GQNNx100x100x100-0-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx100x100x100-1-0" -o "GQNNx100x100x100-1-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx100x100x100-2-0" -o "GQNNx100x100x100-2-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx100x100x100-3-0" -o "GQNNx100x100x100-3-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
bsub -n1 -J "GQNNx100x100x100-4-0" -o "GQNNx100x100x100-4-0.output" "python nn_fitting.py --agent GQNN --hidden_state_sizes 100 100 100 --n_folds 18 --n_ensemble 10 --early_stopping 100 --symmetric False"
sleep .5
