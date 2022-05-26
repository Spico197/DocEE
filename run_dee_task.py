import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
from jinja2 import Template
from loguru import logger
from tqdm import tqdm

from dee.helper import (
    aggregate_task_eval_info,
    print_single_vs_multi_performance,
    print_total_eval_info,
)
from dee.tasks import DEETask, DEETaskSetting
from dee.utils import list_models, set_basic_log_config, strtobool
from print_eval import print_best_test_via_dev, print_detailed_specified_epoch

# set_basic_log_config()


def parse_args(in_args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--task_name", type=str, required=True, help="Take Name")
    arg_parser.add_argument(
        "--data_dir", type=str, default="./Data", help="Data directory"
    )
    arg_parser.add_argument(
        "--exp_dir", type=str, default="./Exps", help="Experiment directory"
    )
    arg_parser.add_argument(
        "--template_filepath",
        type=str,
        default="result_template.html",
        help="Result Template File Path",
    )
    # arg_parser.add_argument('--cuda_visible_devices', type=str, default='',
    #                         help='CUDA_VISIBLE_DEVICES')
    arg_parser.add_argument(
        "--print_final_eval_results",
        type=strtobool,
        default=True,
        help="Whether to print final evaluation results",
    )
    arg_parser.add_argument(
        "--save_cpt_flag",
        type=strtobool,
        default=True,
        help="Whether to save cpt for each epoch",
    )
    arg_parser.add_argument(
        "--skip_train", type=strtobool, default=False, help="Whether to skip training"
    )
    arg_parser.add_argument(
        "--load_dev", type=strtobool, default=True, help="Whether to load dev"
    )
    arg_parser.add_argument(
        "--load_test", type=strtobool, default=True, help="Whether to load test"
    )
    arg_parser.add_argument(
        "--load_inference",
        type=strtobool,
        default=False,
        help="Whether to load inference data",
    )
    arg_parser.add_argument(
        "--inference_epoch",
        type=int,
        default=-1,
        help="which epoch to load for inference",
    )
    arg_parser.add_argument(
        "--run_inference",
        type=strtobool,
        default=False,
        help="Whether to run inference process",
    )
    arg_parser.add_argument(
        "--inference_dump_filepath",
        type=str,
        default="./inference.json",
        help="dumped inference results filepath",
    )
    arg_parser.add_argument(
        "--ensemble", type=strtobool, default=False, help="ensembling"
    )
    arg_parser.add_argument(
        "--ensemble_curr_best_pkl_filepath",
        type=str,
        help="ensemble_curr_best_pkl_filepath",
    )
    arg_parser.add_argument(
        "--ensemble_esmb_best_pkl_filepath",
        type=str,
        help="ensemble_esmb_best_pkl_filepath",
    )
    arg_parser.add_argument(
        "--speed_test", type=strtobool, default=False, help="speed test mode"
    )
    arg_parser.add_argument(
        "--speed_test_epochs", type=int, default=10, help="speed test epoch"
    )
    arg_parser.add_argument(
        "--debug_display", type=strtobool, default=False, help="debug display"
    )
    arg_parser.add_argument(
        "--debug_display_midout_dir",
        type=str,
        help="debug display mid output directory",
    )
    arg_parser.add_argument(
        "--debug_display_epoch", type=int, default=1, help="debug display epoch"
    )
    arg_parser.add_argument(
        "--debug_display_doc_type",
        type=str,
        default="o2o",
        choices=["o2m", "o2o", "m2m", "overall"],
        help="debug display doc type",
    )
    arg_parser.add_argument(
        "--debug_display_span_type",
        type=str,
        default="pred_span",
        choices=["pred_span", "gold_span"],
        help="debug display span type",
    )
    arg_parser.add_argument(
        "--eval_model_names",
        type=str,
        default="DCFEE-O,DCFEE-M,GreedyDec,Doc2EDAG,LSTMMTL,LSTMMTL2CompleteGraph,"
        + ",".join(list_models()),
        help="Models to be evaluated, seperated by ','",
    )
    arg_parser.add_argument(
        "--re_eval_flag",
        type=strtobool,
        default=False,
        help="Whether to re-evaluate previous predictions",
    )
    arg_parser.add_argument(
        "--parallel_decorate",
        action="store_true",
        default=False,
        help="whether to decorate model with parallel setting",
    )

    # add task setting arguments
    for key, val in DEETaskSetting.base_attr_default_pairs:
        if isinstance(val, bool):
            arg_parser.add_argument("--" + key, type=strtobool, default=val)
        else:
            arg_parser.add_argument("--" + key, type=type(val), default=val)

    arg_info = arg_parser.parse_args(args=in_args)

    return arg_info


def render_results(template_filepath, data):
    template_content = open(template_filepath, "rt", encoding="utf-8").read()
    template = Template(template_content)
    rendered_results = template.render(
        task_name=data["task_name"],
        total_results=data["total_results"],
        sm_results=data["sm_results"],
        pred_results=data["pred_results"],
        gold_results=data["gold_results"],
    )
    return rendered_results


if __name__ == "__main__":
    in_argv = parse_args()

    if in_argv.local_rank != -1:
        in_argv.parallel_decorate = True

    task_dir = os.path.join(in_argv.exp_dir, in_argv.task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir, exist_ok=True)

    in_argv.model_dir = os.path.join(task_dir, "Model")
    in_argv.output_dir = os.path.join(task_dir, "Output")
    in_argv.summary_dir_name = os.path.join(task_dir, "Summary/Summary")

    logger.add(os.path.join(task_dir, "log.log"), backtrace=True, diagnose=True)

    # in_argv must contain 'data_dir', 'model_dir', 'output_dir'
    if not in_argv.skip_train:
        dee_setting = DEETaskSetting(**in_argv.__dict__)
    else:
        dee_setting = DEETaskSetting.from_pretrained(
            os.path.join(task_dir, "{}.task_setting.json".format(in_argv.cpt_file_name))
        )
        if in_argv.local_rank == -1 and dee_setting.local_rank != -1:
            dee_setting.local_rank = -1

    dee_setting.filtered_data_types = in_argv.filtered_data_types

    # build task
    dee_task = DEETask(
        dee_setting,
        load_train=not in_argv.skip_train,
        load_dev=in_argv.load_dev,
        load_test=in_argv.load_test,
        load_inference=in_argv.load_inference,
        parallel_decorate=in_argv.parallel_decorate,
    )

    if in_argv.speed_test:
        func_kwargs = dict(
            features=dee_task.test_features,
            use_gold_span=False,
            heuristic_type=None,
        )
        best_epoch = print_best_test_via_dev(
            in_argv.task_name, dee_setting.model_type, dee_setting.num_train_epochs
        )
        dee_task.resume_cpt_at(best_epoch)
        # prepare data loader
        eval_dataloader = dee_task.prepare_data_loader(
            dee_task.test_dataset, in_argv.eval_batch_size, rand_flag=False
        )
        total_time = []
        num_batch = len(eval_dataloader)
        num_docs = len(dee_task.test_dataset)
        for i in range(in_argv.speed_test_epochs):
            # enter eval mode
            if dee_task.model is not None:
                dee_task.model.eval()

            iter_desc = "Evaluation"
            if dee_task.in_distributed_mode():
                iter_desc = "Rank {} {}".format(dist.get_rank(), iter_desc)

            pbar = tqdm(eval_dataloader, desc=iter_desc, ncols=80, ascii=True)
            for step, batch in enumerate(pbar):
                batch = dee_task.set_batch_to_device(batch)

                with torch.no_grad():
                    # this func must run batch_info = model(batch_input)
                    # and metrics is an instance of torch.Tensor with Size([batch_size, ...])
                    # to fit the DataParallel and DistributedParallel functionality
                    start_time = time.time()
                    batch_info = dee_task.get_event_decode_result_on_batch(
                        batch, **func_kwargs
                    )
                    used_time = time.time() - start_time
                    total_time.append(used_time)

            logger.info(pbar)

        logger.info(
            f"Task: {in_argv.task_name}, Model: {dee_setting.model_type}, eval batchsize: {in_argv.eval_batch_size}"
        )
        logger.info(
            f"Speed test: #docs: {num_docs}, #batches: {num_batch}, #eval_epochs: {in_argv.speed_test_epochs}"
        )
        logger.info(
            f"Total used time: {sum(total_time):.3f}, avg time per batch: {statistics.mean(total_time):.3f}"
        )
        logger.info(
            f"Inference speed (docs): {in_argv.speed_test_epochs * num_docs / sum(total_time):.3f} docs/s"
        )

        sys.exit(0)

    if not in_argv.skip_train:
        # dump hyper-parameter settings
        if dee_task.is_master_node():
            fn = "{}.task_setting.json".format(dee_setting.cpt_file_name)
            dee_setting.dump_to(task_dir, file_name=fn)

        dee_task.train(save_cpt_flag=in_argv.save_cpt_flag)

        if dist.is_initialized():
            dist.barrier()
    else:
        dee_task.logging("Skip training")

    if in_argv.run_inference:
        if in_argv.inference_epoch < 0:
            best_epoch = print_best_test_via_dev(
                in_argv.task_name, dee_setting.model_type, dee_setting.num_train_epochs
            )
        else:
            best_epoch = in_argv.inference_epoch
        assert dee_task.inference_dataset is not None
        dee_task.inference(
            resume_epoch=int(best_epoch), dump_filepath=in_argv.inference_dump_filepath
        )

    if in_argv.debug_display:
        # import torch
        # from torch.utils import tensorboard as tb
        # sw = tb.SummaryWriter("runs/biaffine_weight")
        # for i in range(1, 99):
        #     dee_task.resume_cpt_at(i)
        #     sw.add_image("out0", 255 * torch.sigmoid(dee_task.model.biaffine.weight[0].unsqueeze(0)), global_step=i)
        #     sw.add_image("out1", 255 * torch.sigmoid(dee_task.model.biaffine.weight[1].unsqueeze(0)), global_step=i)
        #     sw.add_histogram("biaffine_weight_histogram", dee_task.model.biaffine.weight, global_step=i)
        # sw.flush()
        # sw.close()
        dee_task.debug_display(
            in_argv.debug_display_doc_type,
            in_argv.debug_display_span_type,
            in_argv.debug_display_epoch,
            in_argv.debug_display_midout_dir,
        )
        sys.exit()

    if in_argv.ensemble:
        dee_task.ensemble_dee_prediction(
            in_argv.ensemble_curr_best_pkl_filepath,
            in_argv.ensemble_esmb_best_pkl_filepath,
        )
        sys.exit()

    if in_argv.print_final_eval_results and dee_task.is_master_node():
        """"""
        # dee_task.resume_cpt_at(77)
        # dump_decode_pkl_path = os.path.join(dee_task.setting.output_dir, 'dee_eval.dev.pred_span.TriggerAwarePrunedCompleteGraph.77.pkl')
        # dee_task.base_eval(
        #     dee_task.dev_dataset, DEETask.get_event_decode_result_on_batch,
        #     reduce_info_type='none', dump_pkl_path=dump_decode_pkl_path,
        #     features=dee_task.dev_features, use_gold_span=False, heuristic_type=None,
        # )
        # dump_decode_pkl_path = os.path.join(dee_task.setting.output_dir, 'dee_eval.test.pred_span.TriggerAwarePrunedCompleteGraph.77.pkl')
        # dee_task.base_eval(
        #     dee_task.test_dataset, DEETask.get_event_decode_result_on_batch,
        #     reduce_info_type='none', dump_pkl_path=dump_decode_pkl_path,
        #     features=dee_task.test_features, use_gold_span=False, heuristic_type=None,
        # )
        """"""

        if in_argv.re_eval_flag:
            doc_type2data_span_type2model_str2epoch_res_list = (
                dee_task.reevaluate_dee_prediction(dump_flag=True)
            )
        else:
            doc_type2data_span_type2model_str2epoch_res_list = aggregate_task_eval_info(
                in_argv.output_dir, dump_flag=True
            )
        doc_type = "overall"
        data_type = "test"
        span_type = "pred_span"
        metric_type = "micro"
        mstr_bepoch_list, total_results = print_total_eval_info(
            doc_type2data_span_type2model_str2epoch_res_list,
            dee_task.event_template,
            metric_type=metric_type,
            span_type=span_type,
            model_strs=in_argv.eval_model_names.split(","),
            doc_type=doc_type,
            target_set=data_type,
        )
        sm_results = print_single_vs_multi_performance(
            mstr_bepoch_list,
            in_argv.output_dir,
            dee_task.test_features,
            dee_task.event_template,
            dee_task.setting.event_relevant_combination,
            metric_type=metric_type,
            data_type=data_type,
            span_type=span_type,
        )

        model_types = [x["ModelType"] for x in total_results]
        pred_results = []
        gold_results = []
        for model_type in model_types:
            best_epoch = print_best_test_via_dev(
                in_argv.task_name,
                model_type,
                in_argv.num_train_epochs,
                span_type=span_type,
                data_type=doc_type,
                measure_key="MicroF1" if metric_type == "micro" else "MacroF1",
            )
            pred_result = print_detailed_specified_epoch(
                in_argv.task_name, model_type, best_epoch, span_type="pred_span"
            )
            pred_results.append(pred_result)
            gold_result = print_detailed_specified_epoch(
                in_argv.task_name, model_type, best_epoch, span_type="gold_span"
            )
            gold_results.append(gold_result)

        html_data = dict(
            task_name=in_argv.task_name,
            total_results=total_results,
            sm_results=sm_results,
            pred_results=pred_results,
            gold_results=gold_results,
        )
        if not os.path.exists("./Results/data"):
            os.makedirs("./Results/data")
        with open(
            os.path.join("./Results/data", "data-{}.json".format(in_argv.task_name)),
            "wt",
            encoding="utf-8",
        ) as fout:
            json.dump(html_data, fout, ensure_ascii=False)

        try:
            html_results = render_results(in_argv.template_filepath, data=html_data)
            with open(
                os.path.join(task_dir, "results-{}.html".format(in_argv.task_name)),
                "wt",
                encoding="utf-8",
            ) as fout:
                fout.write(html_results)
            if not os.path.exists("./Results/html"):
                os.makedirs("./Results/html")
            with open(
                os.path.join(
                    "./Results/html", "results-{}.html".format(in_argv.task_name)
                ),
                "wt",
                encoding="utf-8",
            ) as fout:
                fout.write(html_results)
        except Exception:
            pass

    # ensure every processes exit at the same time
    if dist.is_initialized():
        dist.barrier()
