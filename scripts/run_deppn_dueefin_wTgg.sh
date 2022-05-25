#!/bin/bash

set -vx

CUDA="0,1,2,3"
NUM_GPUS=4
TASK_NAME='deppn'
MODEL_NAME='DEPPNModel'

RUN_MODE='dueefin_w_tgg'
TEMPLATE='dueefin_w_tgg'
INFERENCE_DUMPPATH='deppn_duee_fin_w_trigger.json'

{
    #  CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${CUDA} python run_dee_task.py \
    CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/train_multi.sh ${NUM_GPUS} \
        --data_dir='Data/DuEEData' \
        --bert_model='/home/tzhu/bert-pretrained-models/bert-base-chinese' \
        --task_name=${TASK_NAME} \
        --model_type=${MODEL_NAME} \
        --cpt_file_name=${MODEL_NAME} \
        --add_greedy_dec=False \
        --use_bert=False \
        --skip_train=False \
        --run_mode="${RUN_MODE}" \
        --event_type_template="${TEMPLATE}" \
        --resume_latest_cpt=False \
        --deppn_train_nopair_sets=True  \
        --num_train_epochs=100 \
        --train_batch_size=32 \
        --gradient_accumulation_steps=8 \
        --learning_rate=0.0002 \
        --deppn_decoder_lr=0.0001 \
        --num_ner_tf_layers=8 \
        --load_inference=True \
        --inference_epoch=-1 \
        --run_inference=True \
        --filtered_data_types='o2o,o2m,m2m,unk' \
        --inference_dump_filepath="${INFERENCE_DUMPPATH}" \
        --deppn_train_on_multi_events=True \
        --deppn_train_on_single_event=True
}
