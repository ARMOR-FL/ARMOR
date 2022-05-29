#!/usr/bin/env bash

GPU=$1

DATASET=$2

CLIENT_NUM=$3

PERCENT_SUB=$4

ETA=$5

P=$6

python3 ./attack_dmd.py \
 --cuda $GPU \
 --dataset $DATASET \
 --client_num_in_total $CLIENT_NUM \
 --percent_sub $PERCENT_SUB \
 --eta $ETA \
 --p $P