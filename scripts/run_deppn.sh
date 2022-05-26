#!/bin/bash

set -vx

CUDA="0,1,2,3"
NUM_GPUS=4
TASK_NAME='deppn-chfinann'
MODEL_NAME='DEPPNModel'

{
    # CUDA_VISIBLE_DEVICES=${CUDA} python run_dee_task.py \
    CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/train_multi.sh ${NUM_GPUS} \
        --task_name=${TASK_NAME} \
        --model_type=${MODEL_NAME} \
        --cpt_file_name=${MODEL_NAME} \
        --use_bert=False \
        --skip_train=False \
        --run_mode='full' \
        --add_greedy_dec=False \
        --resume_latest_cpt=False \
        --deppn_train_nopair_sets=True  \
        --num_train_epochs=100 \
        --train_batch_size=32 \
        --gradient_accumulation_steps=8 \
        --learning_rate=0.0002 \
        --deppn_decoder_lr=0.0001 \
        --num_ner_tf_layers=8 \
        --deppn_train_on_multi_events=True \
        --deppn_train_on_single_event=True
}
