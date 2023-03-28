#!/bin/bash

{
    MODEL_NAME='TriggerAwarePrunedCompleteGraph'
    TASK_NAME='PTPCG_CCKS2020'
    echo "('${TASK_NAME}', '${MODEL_NAME}'),    # $(date)" >> RECORDS.md
    echo "Task Name: $TASK_NAME"
    echo "Model Name: $MODEL_NAME"

    # GPU_SCOPE="0,1,2,3"
    # REQ_GPU_NUM=1
    GPUS="4"
    # GPUS=$(python wait.py --task_name="$TASK_NAME" --cuda=$GPU_SCOPE --wait="schedule" --req_gpu_num=$REQ_GPU_NUM)
    echo "GPUS: $GPUS"
    EPOCH_NUM=50

    if [[ -z "$GPUS" ]]; then
        echo "GPUS is empty, stop..."
        # python send_message.py "Task $TASK_NAME not started due to empty gpu assigning, please check the log."
        echo "Task $TASK_NAME not started due to empty gpu assigning, please check the log."
    else
        echo "GPU ready."
        # python send_message.py "Task $TASK_NAME started."
        echo "Task $TASK_NAME started."
        CUDA_VISIBLE_DEVICES=${GPUS} python -u run_dee_task.py \
            --data_dir='Data/CCKS2020' \
            --use_bert=True \
            --bert_model='/data/tzhu/PLM/bert-base-chinese' \
            --seed=99 \
            --task_name=${TASK_NAME} \
            --model_type=${MODEL_NAME} \
            --cpt_file_name=${MODEL_NAME} \
            --save_cpt_flag=False \
            --save_best_cpt=True \
            --remove_last_cpt=True \
            --resume_latest_cpt=False \
            --optimizer='adam' \
            --learning_rate=0.00005 \
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
            --schedule_epoch_start=5 \
            --schedule_epoch_length=5 \
            --num_train_epochs=${EPOCH_NUM} \
            --run_mode='full' \
            --filtered_data_types='o2o,o2m,m2m' \
            --skip_train=False \
            --re_eval_flag=False \
            --add_greedy_dec=False \
            --num_lstm_layers=2 \
            --hidden_size=768 \
            --biaffine_hidden_size=512 \
            --biaffine_hard_threshold=0.5 \
            --at_least_one_comb=True \
            --include_complementary_ents=True \
            --event_type_template='ccks2020' \
            --use_span_lstm=True \
            --span_lstm_num_layer=2 \
            --role_by_encoding=True \
            --use_token_role=True \
            --ment_feature_type='concat' \
            --ment_type_hidden_size=32
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
