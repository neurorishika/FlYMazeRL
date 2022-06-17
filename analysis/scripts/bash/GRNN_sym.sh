#!/bin/bash
RESERVOIR_SIZE=2

python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size $RESERVOIR_SIZE --n_folds 18 --n_ensemble 10 --early_stopping 100 &
python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size $RESERVOIR_SIZE --n_folds 18 --n_ensemble 10 --early_stopping 100 &
python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size $RESERVOIR_SIZE --n_folds 18 --n_ensemble 10 --early_stopping 100 &
python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size $RESERVOIR_SIZE --n_folds 18 --n_ensemble 10 --early_stopping 100 &
python nn_fitting.py --agent GRNN --num_reservoir 1 --reservoir_size $RESERVOIR_SIZE --n_folds 18 --n_ensemble 10 --early_stopping 100 &
wait