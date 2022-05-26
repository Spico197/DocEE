.. toctree::
   :maxdepth: 2
   :caption: Contents:


Inference
=========

:Authors:
    Tong Zhu

:Last Update: May 26th, 2022


DocEE provides convenient inference APIs for generating submitting results (like DuEE-fin) or make predictions with plain texts as input.

Here, we introduce the inference usages.

Generating Results to Submit Online Platforms
---------------------------------------------

Since the DuEE-fin dataset needs to be evaluated on the online test set, a submit file must be generated offline in advance.

To make inference results on DuEE-fin, make sure you have set the correct ``run_mode`` in settings.
If you use our provided scripts, this should not be a problem.
Please be aware that ``run_mode`` is different from templates.
It is used for identifying file names for different data.

There are some arguments that relates to inference:
- ``run_inference``: whether to make inference
- ``load_inference``: whether to load inference dataset
- ``inference_epoch``: which epoch is selected to load model parameters. ``-1`` to use the best model.
- ``inference_dump_filepath``: the result file path
- ``filtered_data_types``: ``unk`` must be included in it to load inference dataset. i.e. ``o2o,o2m,m2m,unk``

For example, you may want to make inference like:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 python -u run_dee_task.py \
        --data_dir='Data/DuEEData' \
        --task_name='dueefin_w_tgg' \
        --model_type=${MODEL_NAME} \
        --cpt_file_name=${MODEL_NAME} \
        --eval_batch_size=16 \
        --run_mode='dueefin_w_tgg' \
        --filtered_data_types='o2o,o2m,m2m,unk' \
        --skip_train=True \
        --load_dev=False \
        --load_test=False \
        --load_inference=True \
        --inference_epoch=-1 \
        --run_inference=True \
        --inference_dump_filepath='dueefin_t1_submit_new.json' \
        --add_greedy_dec=False

After obtaining the result file ``dueefin_t1_submit_new.json``, you may want to remove redundant information via the ``dueefin_post_process.py`` script.
Set ``to_remove_filepath`` and ``save_filepath`` items in the script, and run the following command in shell to get the final file.

.. code-block:: bash

    python dueefin_post_process.py


Make Predictions from Plain Texts
---------------------------------

Yes! DocEE provides a simple API to predict event instances.

Basically, you need to load a trained model parameter at the beginning, for example, ``dee_task.resume_cpt_at(epoch_idx)``.
Then you could use ``dee_task.predict_one(doc_string)`` to obtain the predicted event instances.

Check ``inference.py`` for the detailed demo usage.
