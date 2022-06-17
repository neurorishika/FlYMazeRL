#!/bin/bash
RESERVOIR_SIZE=2

python nn_fitting.py --agent GRNN --reservoir_size $RESERVOIR_SIZE --num_reservoir 1 --n_ensemble 10 --n_folds 18 &
python nn_fitting.py --agent GRNN --reservoir_size $RESERVOIR_SIZE --num_reservoir 1 --n_ensemble 10 --n_folds 18 &
python nn_fitting.py --agent GRNN --reservoir_size $RESERVOIR_SIZE --num_reservoir 1 --n_ensemble 10 --n_folds 18 &
python nn_fitting.py --agent GRNN --reservoir_size $RESERVOIR_SIZE --num_reservoir 1 --n_ensemble 10 --n_folds 18 &
python nn_fitting.py --agent GRNN --reservoir_size $RESERVOIR_SIZE --num_reservoir 1 --n_ensemble 10 --n_folds 18 &
wait