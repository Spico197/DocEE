#! /bin/bash

NUM_GPUS=$1
shift

python -m torch.distributed.launch --master_port=25662 --nproc_per_node ${NUM_GPUS} run_dee_task.py $*
