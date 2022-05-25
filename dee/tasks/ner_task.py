import json
import os
from collections import defaultdict

from loguru import logger

from dee.event_types import get_event_template
from dee.helper.ner import (
    NERExample,
    NERExampleLoader,
    NERFeatureConverter,
    convert_ner_features_to_dataset,
)
from dee.modules import BertForBasicNER, judge_ner_prediction
from dee.tasks.base_task import BasePytorchTask, TaskSetting
from dee.utils import EPS, BertTokenizerForDocEE, default_dump_json


class NERTaskSetting(TaskSetting):
    def __init__(self, **kwargs):
        ner_key_attrs = []
        ner_attr_default_pairs = [
            ("bert_model", "bert-base-chinese"),
            ("train_file_name", "train.json"),
            ("dev_file_name", "dev.json"),
            ("test_file_name", "test.json"),
            ("max_seq_len", 128),
            ("train_batch_size", 32),
            ("eval_batch_size", 256),
            ("learning_rate", 2e-5),
            ("num_train_epochs", 3.0),
            ("warmup_proportion", 0.1),
            ("no_cuda", False),
            ("local_rank", -1),
            ("seed", 99),
            ("gradient_accumulation_steps", 1),
            ("optimize_on_cpu", True),
            ("fp16", False),
            ("loss_scale", 128),
            ("cpt_file_name", "ner_task.cpt"),
            ("summary_dir_name", "/tmp/summary"),
        ]
        super(NERTaskSetting, self).__init__(
            ner_key_attrs, ner_attr_default_pairs, **kwargs
        )


class NERTask(BasePytorchTask):
    """Named Entity Recognition Task"""

    def __init__(
        self,
        setting,
        load_train=True,
        load_dev=True,
        load_test=True,
        build_model=True,
        parallel_decorate=True,
        resume_model=False,
        resume_optimizer=False,
    ):
        super(NERTask, self).__init__(setting)
        self.logger = logger
        self.logging("Initializing {}".format(self.__class__.__name__))

        self.event_template = get_event_template(setting.event_type_template)
        # initialize entity label list
        self.entity_label_list = NERExample.get_entity_label_list(self.event_template)
        # initialize tokenizer
        self.tokenizer = BertTokenizerForDocEE.from_pretrained(
            self.setting.bert_model, doc_lang=self.setting.doc_lang
        )
        # initialize feature converter
        self.feature_converter_func = NERFeatureConverter(
            self.entity_label_list, self.setting.max_seq_len, self.tokenizer
        )
        self.example_loader = NERExampleLoader(self.tokenizer)

        # load data
        self._load_data(
            self.example_loader.load_ner_dataset,
            self.feature_converter_func,
            convert_ner_features_to_dataset,
            load_train=load_train,
            load_dev=load_dev,
            load_test=load_test,
        )

        # build model
        if build_model:
            self.model = BertForBasicNER.from_pretrained(
                self.setting.bert_model, len(self.entity_label_list)
            )
            self.setting.update_by_dict(
                self.model.config.__dict__
            )  # BertConfig dictionary
            self._decorate_model(parallel_decorate=parallel_decorate)

        # prepare optimizer
        if build_model and load_train:
            self._init_bert_optimizer()

        # resume option
        if build_model and (resume_model or resume_optimizer):
            self.resume_checkpoint(
                resume_model=resume_model, resume_optimizer=resume_optimizer
            )

        self.logging("Successfully initialize {}".format(self.__class__.__name__))

    def reload_data(self, data_type="return", file_name=None, file_path=None, **kwargs):
        """Either file_name or file_path needs to be provided,
        data_type: return (default), return (examples, features, dataset)
                   train, override self.train_xxx
                   dev, override self.dev_xxx
                   test, override self.test_xxx
        """
        return super(NERTask, self).reload_data(
            self.example_loader.load_ner_dataset,
            self.feature_converter_func,
            convert_ner_features_to_dataset,
            data_type=data_type,
            file_name=file_name,
            file_path=file_path,
        )

    def train(self):
        self.logging("=" * 20 + "Start Training" + "=" * 20)
        self.base_train(get_ner_loss_on_batch)

    def eval(self, eval_dataset, eval_save_prefix="", pgm_return_flag=False):
        self.logging("=" * 20 + "Start Evaluation" + "=" * 20)
        # 1. get total prediction info
        # pgm denotes (pred_label, gold_label, token_mask)
        # size = [num_examples, max_seq_len, 3]
        # value = [[(pred_label, gold_label, token_mask), ...], ...]
        total_seq_pgm = self.get_total_prediction(eval_dataset)
        num_examples, max_seq_len, _ = total_seq_pgm.size()

        # 2. collect per-entity-label tp, fp, fn counts
        ent_lid2tp_cnt = defaultdict(lambda: 0)
        ent_lid2fp_cnt = defaultdict(lambda: 0)
        ent_lid2fn_cnt = defaultdict(lambda: 0)
        for bid in range(num_examples):
            seq_pgm = total_seq_pgm[bid]  # [max_seq_len, 3]
            seq_pred = seq_pgm[:, 0]  # [max_seq_len]
            seq_gold = seq_pgm[:, 1]
            seq_mask = seq_pgm[:, 2]

            seq_pred_lid = seq_pred[seq_mask == 1]  # [seq_len]
            seq_gold_lid = seq_gold[seq_mask == 1]
            ner_tp_set, ner_fp_set, ner_fn_set = judge_ner_prediction(
                seq_pred_lid, seq_gold_lid
            )
            for ent_lid2cnt, ex_ner_set in [
                (ent_lid2tp_cnt, ner_tp_set),
                (ent_lid2fp_cnt, ner_fp_set),
                (ent_lid2fn_cnt, ner_fn_set),
            ]:
                for ent_idx_s, ent_idx_e, ent_lid in ex_ner_set:
                    ent_lid2cnt[ent_lid] += 1

        # 3. calculate per-entity-label metrics and collect global counts
        ent_label_eval_infos = []
        g_ner_tp_cnt = 0
        g_ner_fp_cnt = 0
        g_ner_fn_cnt = 0
        # Entity Label Id, 0 for others, odd for BEGIN-ENTITY, even for INSIDE-ENTITY
        # using odd is enough to represent the entity type
        for ent_lid in range(1, len(self.entity_label_list), 2):
            el_name = self.entity_label_list[ent_lid]
            el_tp_cnt, el_fp_cnt, el_fn_cnt = (
                ent_lid2tp_cnt[ent_lid],
                ent_lid2fp_cnt[ent_lid],
                ent_lid2fn_cnt[ent_lid],
            )

            el_pred_cnt = el_tp_cnt + el_fp_cnt
            el_gold_cnt = el_tp_cnt + el_fn_cnt
            el_prec = el_tp_cnt / el_pred_cnt if el_pred_cnt > 0 else 0
            el_recall = el_tp_cnt / el_gold_cnt if el_gold_cnt > 0 else 0
            el_f1 = (
                2 / (1 / el_prec + 1 / el_recall)
                if el_prec > EPS and el_recall > EPS
                else 0
            )

            # per-entity-label evaluation info
            el_eval_info = {
                "entity_label_indexes": (ent_lid, ent_lid + 1),
                "entity_label": el_name[2:],  # omit 'B-' prefix
                "ner_tp_cnt": el_tp_cnt,
                "ner_fp_cnt": el_fp_cnt,
                "ner_fn_cnt": el_fn_cnt,
                "ner_prec": el_prec,
                "ner_recall": el_recall,
                "ner_f1": el_f1,
            }
            ent_label_eval_infos.append(el_eval_info)

            # collect global count info
            g_ner_tp_cnt += el_tp_cnt
            g_ner_fp_cnt += el_fp_cnt
            g_ner_fn_cnt += el_fn_cnt

        # 4. summarize total evaluation info
        g_ner_pred_cnt = g_ner_tp_cnt + g_ner_fp_cnt
        g_ner_gold_cnt = g_ner_tp_cnt + g_ner_fn_cnt
        g_ner_prec = g_ner_tp_cnt / g_ner_pred_cnt if g_ner_pred_cnt > 0 else 0
        g_ner_recall = g_ner_tp_cnt / g_ner_gold_cnt if g_ner_gold_cnt > 0 else 0
        g_ner_f1 = (
            2 / (1 / g_ner_prec + 1 / g_ner_recall)
            if g_ner_prec > EPS and g_ner_recall > EPS
            else 0
        )

        total_eval_info = {
            "eval_name": eval_save_prefix,
            "num_examples": num_examples,
            "ner_tp_cnt": g_ner_tp_cnt,
            "ner_fp_cnt": g_ner_fp_cnt,
            "ner_fn_cnt": g_ner_fn_cnt,
            "ner_prec": g_ner_prec,
            "ner_recall": g_ner_recall,
            "ner_f1": g_ner_f1,
            "per_ent_label_eval": ent_label_eval_infos,
        }

        self.logging(
            "Evaluation Results\n{:.300s} ...".format(
                json.dumps(total_eval_info, indent=4)
            )
        )

        if eval_save_prefix:
            eval_res_fp = os.path.join(
                self.setting.output_dir, "{}.eval".format(eval_save_prefix)
            )
            self.logging("Dump eval results into {}".format(eval_res_fp))
            default_dump_json(total_eval_info, eval_res_fp)

        if pgm_return_flag:
            return total_seq_pgm
        else:
            return total_eval_info

    def get_total_prediction(self, eval_dataset):
        self.logging("=" * 20 + "Get Total Prediction" + "=" * 20)
        total_pred_gold_mask = self.base_eval(
            eval_dataset, get_ner_pred_on_batch, reduce_info_type="none"
        )
        # torch.Tensor(dtype=torch.long, device='cpu')
        # size = [batch_size, seq_len, 3]
        # value = [[(pred_label, gold_label, token_mask), ...], ...]
        return total_pred_gold_mask


def normalize_batch_seq_len(input_seq_lens, *batch_seq_tensors):
    batch_max_seq_len = input_seq_lens.max().item()
    normed_tensors = []
    for batch_seq_tensor in batch_seq_tensors:
        if batch_seq_tensor.dim() == 2:
            normed_tensors.append(batch_seq_tensor[:, :batch_max_seq_len])
        elif batch_seq_tensor.dim() == 1:
            normed_tensors.append(batch_seq_tensor)
        else:
            raise Exception(
                "Unsupported batch_seq_tensor dimension {}".format(
                    batch_seq_tensor.dim()
                )
            )

    return normed_tensors


def prepare_ner_batch(batch, resize_len=True):
    # prepare batch
    input_ids, input_masks, segment_ids, label_ids, input_lens = batch
    if resize_len:
        input_ids, input_masks, segment_ids, label_ids = normalize_batch_seq_len(
            input_lens, input_ids, input_masks, segment_ids, label_ids
        )

    return input_ids, input_masks, segment_ids, label_ids


def get_ner_loss_on_batch(ner_task, batch):
    input_ids, input_masks, segment_ids, label_ids = prepare_ner_batch(
        batch, resize_len=True
    )
    loss, _ = ner_task.model(
        input_ids, input_masks, token_type_ids=segment_ids, label_ids=label_ids
    )

    return loss


def get_ner_metrics_on_batch(ner_task, batch):
    input_ids, input_masks, segment_ids, label_ids = prepare_ner_batch(
        batch, resize_len=True
    )
    batch_metrics = ner_task.model(
        input_ids,
        input_masks,
        token_type_ids=segment_ids,
        label_ids=label_ids,
        eval_flag=True,
        eval_for_metric=True,
    )

    return batch_metrics


def get_ner_pred_on_batch(ner_task, batch):
    # important to set resize_len to False to maintain the same seq len between batches
    input_ids, input_masks, segment_ids, label_ids = prepare_ner_batch(
        batch, resize_len=False
    )
    batch_seq_pred_gold_mask = ner_task.model(
        input_ids,
        input_masks,
        token_type_ids=segment_ids,
        label_ids=label_ids,
        eval_flag=True,
        eval_for_metric=False,
    )
    # size = [batch_size, max_seq_len, 3]
    # value = [[(pred_label, gold_label, token_mask), ...], ...]
    return batch_seq_pred_gold_mask
