#!/bin/bash

{
    MODEL_NAME='TriggerAwarePrunedCompleteGraph'
    TASK_NAME='PTPCG_P1-DuEE_fin-woTgg-wOtherType-midImpt'
    RUN_MODE='dueefin_wo_tgg'
    TEMPLATE='dueefin_wo_tgg_mid_importance'
    INFERENCE_DUMPPATH='dueefin_PTPCG_woTgg_midImpt.json'
    echo "('${TASK_NAME}', '${MODEL_NAME}'),    # $(date)" >> RECORDS.md
    echo "Task Name: $TASK_NAME"
    echo "Model Name: $MODEL_NAME"

    # GPU_SCOPE="0,1,2,3"
    # REQ_GPU_NUM=1
    GPUS="3"
    # GPUS=$(python wait.py --task_name="$TASK_NAME" --cuda=$GPU_SCOPE --wait="schedule" --req_gpu_num=$REQ_GPU_NUM)
    echo "GPUS: $GPUS"
    EPOCH_NUM=100

    if [[ -z "$GPUS" ]]; then
        echo "GPUS is empty, stop..."
        # python send_message.py "Task $TASK_NAME not started due to empty gpu assigning, please check the log."
        echo "Task $TASK_NAME not started due to empty gpu assigning, please check the log."
    else
        echo "GPU ready."
        # python send_message.py "Task $TASK_NAME started."
        echo "Task $TASK_NAME started."
        CUDA_VISIBLE_DEVICES=${GPUS} python -u run_dee_task.py \
            --use_bert=False \
            --bert_model='/home/tzhu/bert-pretrained-models/bert-base-chinese' \
            --seed=99 \
            --data_dir='Data/DuEEData' \
            --task_name=${TASK_NAME} \
            --model_type=${MODEL_NAME} \
            --cpt_file_name=${MODEL_NAME} \
            --save_cpt_flag=False \
            --save_best_cpt=True \
            --remove_last_cpt=True \
            --optimizer='adam' \
            --learning_rate=0.0005 \
            --dropout=0.1 \
            --gradient_accumulation_steps=8 \
            --train_batch_size=64 \
            --eval_batch_size=16 \
            --max_clique_decode=True \
            --num_triggers=1 \
            --eval_num_triggers=1 \
            --with_left_trigger=True \
            --directed_trigger_graph=True \
            --use_scheduled_sampling=True \
            --schedule_epoch_start=10 \
            --schedule_epoch_length=10 \
            --num_train_epochs=${EPOCH_NUM} \
            --run_mode=${RUN_MODE} \
            --filtered_data_types='o2o,o2m,m2m,unk' \
            --skip_train=False \
            --load_dev=True \
            --load_test=True \
            --load_inference=True \
            --inference_epoch=-1 \
            --run_inference=True \
            --inference_dump_filepath=${INFERENCE_DUMPPATH} \
            --re_eval_flag=False \
            --add_greedy_dec=False \
            --num_lstm_layers=2 \
            --hidden_size=768 \
            --biaffine_hidden_size=512 \
            --biaffine_hard_threshold=0.5 \
            --at_least_one_comb=True \
            --include_complementary_ents=True \
            --event_type_template=${TEMPLATE} \
            --use_span_lstm=True \
            --span_lstm_num_layer=2 \
            --role_by_encoding=True \
            --use_token_role=True \
            --ment_feature_type='concat' \
            --ment_type_hidden_size=32

        # # run on inference dataset
        # CUDA_VISIBLE_DEVICES=${GPUS} python -u run_dee_task.py \
        #     --data_dir='Data/DuEEData' \
        #     --task_name=${TASK_NAME} \
        #     --model_type=${MODEL_NAME} \
        #     --cpt_file_name=${MODEL_NAME} \
        #     --eval_batch_size=16 \
        #     --run_mode='dueefin_wo_tgg' \
        #     --filtered_data_types='o2o,o2m,m2m,unk' \
        #     --skip_train=True \
        #     --load_dev=False \
        #     --load_test=False \
        #     --load_inference=True \
        #     --inference_epoch=-1 \
        #     --run_inference=True \
        #     --inference_dump_filepath='dueefin_p1_submit_new.json' \
        #     --add_greedy_dec=False
    fi

    # check if the process has finished normally
    LOG_FILE="Logs/$TASK_NAME.log"
    if [[ -f "$LOG_FILE" ]]; then
        echo "$LOG_FILE exists."
        MATCHED=$(grep 'Combination' "$LOG_FILE")
        if [[ -z "$MATCHED" ]]; then
            # python send_message.py " Something's wrong in task $TASK_NAME, please check the log."
            echo " Something's wrong in task $TASK_NAME, please check the log."
        else
            # python send_message.py --send_result --task_name ${TASK_NAME} --model_name ${MODEL_NAME} --max_epoch ${EPOCH_NUM} "Task $TASK_NAME finished."
            echo "Task $TASK_NAME finished."
        fi
    else
        echo "$LOG_FILE not found."
        # python send_message.py "Task $TASK_NAME finished, but log is not found."
        echo "Task $TASK_NAME finished, but log is not found."
    fi

    exit
}
