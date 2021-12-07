#!/bin/bash

{
    MODEL_NAME='TriggerAwarePrunedCompleteGraph'
    TASK_NAME='n143-Tp44CG-with_left_trigger-comp_ents-OtherType-bs64_8'
    echo "Task Name: $TASK_NAME"
    echo "Model Name: $MODEL_NAME"

    GPU_SCOPE="1,2,3"
    REQ_GPU_NUM=1
    GPUS="0"
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
            --task_name=${TASK_NAME} \
            --model_type=${MODEL_NAME} \
            --cpt_file_name=${MODEL_NAME} \
            --speed_test=True \
            --speed_test_epochs=5 \
            --eval_batch_size=2 \
            --run_mode='full' \
            --filtered_data_types='o2o,o2m,m2m' \
            --skip_train=True \
            --load_dev=False \
            --re_eval_flag=False \
            --add_greedy_dec=False
    fi

    exit
}
