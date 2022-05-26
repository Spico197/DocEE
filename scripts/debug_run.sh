#!/bin/bash

set -vx

{
    MODEL_NAME='TriggerAwarePrunedCompleteGraph'

    CUDA_VISIBLE_DEVICES=3 python -u run_dee_task.py \
        --bert_model='/home/tzhu/bert-pretrained-models/bert-base-chinese' \
        --task_name='debug' \
        --model_type=${MODEL_NAME} \
        --cpt_file_name=${MODEL_NAME} \
        --optimizer='adam' \
        --num_triggers=0 \
        --directed_trigger_graph=False \
        --use_shared_dropout_proj=False \
        --use_layer_norm_b4_biaffine=False \
        --remove_mention_type_layer_norm=False \
        --gradient_accumulation_steps=1 \
        --ent_fix_mode='n' \
        --train_batch_size=1 \
        --use_biaffine_ner=False \
        --span_mention_sum=False \
        --eval_batch_size=1 \
        --resume_latest_cpt=False \
        --num_train_epochs=3 \
        --run_mode='debug' \
        --use_lr_scheduler=False \
        --filtered_data_types='o2o,o2m,m2m' \
        --skip_train=False \
        --re_eval_flag=False \
        --add_greedy_dec=False \
        --learning_rate=0.001 \
        --combination_loss_weight=1.0 \
        --event_relevant_combination=False \
        --use_scheduled_sampling=True \
        --schedule_epoch_start=1 \
        --schedule_epoch_length=1 \
        --num_lstm_layers=2 \
        --biaffine_hidden_size=512 \
        --drop_irr_ents=False \
        --use_span_lstm=True \
        --span_lstm_num_layer=1 \
        --use_span_att=True \
        --span_att_heads=1 \
        --use_biaffine_ner=False \
        --use_masked_crf=False \
        --use_span_lstm_projection=False

    exit
}
