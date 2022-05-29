#!/usr/bin/env bash

GPU=$1

DATASET=$2

CLIENT_NUM=$3

python3 ./fed_train.py \
 --cuda $GPU \
 --dataset $DATASET \
 --client_num_in_total $CLIENT_NUM