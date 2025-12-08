#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}


export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --deterministic ${@:3}