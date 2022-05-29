#!/usr/bin/env bash

DATASET=$1

CLIENT_NUM=$2

PERCENT_SUB=$3

ETA=$4

P=$5

python3 ./process_exp.py \
 --dataset $DATASET \
 --client_num_in_total $CLIENT_NUM \
 --percent_sub $PERCENT_SUB \
 --eta $ETA \
 --p $P