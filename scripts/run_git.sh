TASK_NAME='GIT_reproduction'
CUDA='0,1,2,3'
NUM_GPU=4
MODEL_NAME='GITModel'
# python wait.py --task_name=${TASK_NAME} --wait='queue' --cuda=${CUDA}
CUDA_VISIBLE_DEVICES=${CUDA} ./scripts/train_multi.sh ${NUM_GPU} --task_name ${TASK_NAME}\
--model_type=${MODEL_NAME} \
--cpt_file_name=${MODEL_NAME} \
--add_greedy_dec=False \
--bert_model='bert-base-chinese' \
--ner_num_tf_layers=8 \
--gradient_accumulation_steps=16 \
--train_batch_size=64 \
--eval_batch_size=2 \
--resume_latest_cpt=False \
--num_train_epochs=100 \
--run_mode='full' \
--skip_train=False \
--parallel_decorate
