import copy
import glob
import logging
import os
from itertools import combinations, product
from typing import List, Union

import torch
import torch.distributed as dist
import torch.optim as optim
from loguru import logger
from tqdm import tqdm
from transformers.models.bert.modeling_bert import BertConfig

import dee.models
from dee.event_types import get_event_template
from dee.helper import (
    DEEArgRelFeatureConverter,
    DEEExample,
    DEEExampleLoader,
    DEEFeatureConverter,
    DEPPNFeatureConverter,
    convert_dee_arg_rel_features_to_dataset,
    convert_dee_features_to_dataset,
    convert_deppn_features_to_dataset,
    convert_string_to_raw_input,
    decode_dump_template,
    eval_dump_template,
    match_arg,
    measure_dee_prediction,
    prepare_doc_batch_dict,
)
from dee.models import DCFEEModel, Doc2EDAGModel
from dee.modules import LSTMBiaffineNERModel, LSTMMaskedCRFNERModel
from dee.modules.ner_model import BERTCRFNERModel
from dee.tasks.base_task import BasePytorchTask, TaskSetting
from dee.utils import (
    BertTokenizerForDocEE,
    chain_prod,
    default_dump_json,
    default_load_pkl,
    get_cosine_schedule_with_warmup,
    list_models,
    remove_event_obj_roles,
)


class DEETaskSetting(TaskSetting):
    base_key_attrs = TaskSetting.base_key_attrs
    base_attr_default_pairs = [
        # ('train_file_name', 'typed_train.json'),
        ("train_file_name", "typed_train.json"),
        ("dev_file_name", "typed_dev.json"),
        ("test_file_name", "typed_test.json"),
        # ('train_file_name', 'typed_sample_train_48.json'),
        # ('dev_file_name', 'typed_sample_train_48.json'),
        # ('test_file_name', 'typed_sample_train_48.json'),
        ("summary_dir_name", "Summary/Summary"),
        ("event_type_template", "zheng2019_trigger_graph_no_OtherType"),
        ("max_sent_len", 128),
        ("max_sent_num", 64),
        ("train_batch_size", 64),
        ("gradient_accumulation_steps", 8),
        ("eval_batch_size", 2),
        ("learning_rate", 1e-4),
        ("use_lr_scheduler", False),
        ("lr_scheduler_step", 20),
        ("num_train_epochs", 100),
        # ('num_train_epochs', 30),
        ("no_cuda", False),
        ("local_rank", -1),
        ("seed", 99),
        ("optimize_on_cpu", False),
        ("fp16", False),
        ("use_bert", False),  # whether to use bert as the encoder
        ("use_biaffine_ner", False),  # use biaffine ner model
        ("use_masked_crf", False),
        (
            "bert_model",
            "/home/tzhu/bert-pretrained-models/bert-base-chinese",
        ),  # use which pretrained bert model
        ("only_master_logging", True),  # whether to print logs from multiple processes
        (
            "resume_latest_cpt",
            True,
        ),  # whether to resume latest checkpoints when training for fault tolerance
        ("remove_last_cpt", False),
        ("save_best_cpt", False),
        (
            "cpt_file_name",
            "Doc2EDAG",
        ),  # decide the identity of checkpoints, evaluation results, etc.
        ("model_type", "Doc2EDAG"),  # decide the model class used
        ("rearrange_sent", False),  # whether to rearrange sentences
        ("use_crf_layer", True),  # whether to use CRF Layer
        ("min_teacher_prob", 0.1),  # the minimum prob to use gold spans
        ("schedule_epoch_start", 10),  # from which epoch the scheduled sampling starts
        (
            "schedule_epoch_length",
            10,
        ),  # the number of epochs to linearly transit to the min_teacher_prob
        ("loss_lambda", 0.05),  # the proportion of ner loss
        ("loss_gamma", 1.0),  # the scaling proportion of missed span sentence ner loss
        ("deppn_ner_loss_weight", 0.1),  # the proportion of ner loss
        (
            "deppn_type_loss_weight",
            0.4,
        ),  # the proportion of event type classification loss
        (
            "deppn_event_generation_loss_weight",
            0.5,
        ),  # the proportion of event generation loss
        ("deppn_decoder_lr", 2e-5),  # learning rate of DEPPN decoder
        ("deppn_num_event2role_decoder_layer", 4),
        ("deppn_train_on_multi_events", True),
        ("deppn_train_on_single_event", True),
        ("deppn_event_type_classes", 2),
        ("deppn_num_generated_sets", 5),
        ("deppn_num_set_decoder_layers", 2),
        ("deppn_num_role_decoder_layers", 4),
        ("deppn_cost_weight", {"event_type": 1, "role": 0.5}),
        ("deppn_train_on_multi_roles", False),
        ("deppn_use_event_type_enc", True),
        ("deppn_use_role_decoder", True),
        ("deppn_use_sent_span_encoder", False),
        ("deppn_train_nopair_sets", True),
        ("deppn_hidden_dropout", 0.1),
        ("deppn_layer_norm_eps", 1e-12),
        ("deppn_event_type_weight", [1, 0.2]),
        ("add_greedy_dec", True),  # whether to add additional greedy decoding
        ("use_token_role", True),  # whether to use detailed token role
        (
            "seq_reduce_type",
            "MaxPooling",
        ),  # use 'MaxPooling', 'MeanPooling' or 'AWA' to reduce a tensor sequence
        # network parameters (follow Bert Base)
        ("hidden_size", 768),
        ("dropout", 0.1),
        ("ff_size", 1024),  # feed-forward mid layer size
        ("num_tf_layers", 4),  # transformer layer number
        # ablation study parameters,
        ("use_path_mem", True),  # whether to use the memory module when expanding paths
        ("use_scheduled_sampling", True),  # whether to use the scheduled sampling
        ("use_doc_enc", True),  # whether to use document-level entity encoding
        ("neg_field_loss_scaling", 3.0),  # prefer FNs over FPs
        ("gcn_layer", 3),  # prefer FNs over FPs
        ("num_ner_tf_layers", 4),
        # LSTM MTL
        ("num_lstm_layers", 1),  # number of lstm layers
        ("use_span_lstm", False),  # add lstm module after span representation
        ("span_lstm_num_layer", 1),  # add lstm module after span representation
        ("use_span_att", False),  # add self-attention for spans after lstm encoding
        ("span_att_heads", 4),
        # number of head in dot attention
        ("dot_att_head", 4),
        # comb sampling parameters
        ("comb_samp_min_num_span", 2),
        ("comb_samp_num_samp", 100),
        ("comb_samp_max_samp_times", 1000),
        # Arg Triangle Relation
        # use lstm encoder for spans instead of linear projection before biaffine prediction
        ("use_span_lstm_projection", False),
        ("biaffine_hidden_size", 256),
        ("triaffine_hidden_size", 150),
        ("vi_max_iter", 3),
        ("biaffine_hard_threshold", 0.5),
        ("event_cls_loss_weight", 1.0),
        ("smooth_attn_loss_weight", 1.0),
        # ('combination_loss_weight', 0.1),
        ("combination_loss_weight", 1.0),
        ("comb_cls_loss_weight", 1.0),
        ("comb_sim_loss_weight", 1.0),
        ("span_cls_loss_weight", 1.0),
        ("use_comb_cls_pred", False),
        ("role_loss_weight", 1.0),
        ("event_relevant_combination", False),
        # running mode for data selection and other debug options
        # choices: full, quarter, debug
        # full: all the training data
        # quarter: use quarter of training data
        # debug: use the 48 debug instances and simplify
        #        the pred_adj_mat decoding for CompleteGraph model
        ("run_mode", "full"),
        # drop irrelevant entities during data preprocessing and
        # make sure all the entities appear in the final event combinations
        ("drop_irr_ents", False),
        ("at_least_one_comb", False),
        ("include_complementary_ents", False),
        ("filtered_data_types", "o2o,o2m,m2m"),
        ("ent_context_window", 20),
        ("biaffine_grad_clip", False),
        ("global_grad_clip", False),
        # entity fixing mode:
        # - `n`: no fixing
        # - `-`: remove wrong ones
        # - `f`: fix wrong ones
        ("ent_fix_mode", "n"),
        ("span_mention_sum", False),
        ("add_adj_mat_weight_bias", False),
        ("optimizer", "adam"),
        # number of triggers, choices among 1, 2, 3 and the others (complete graph)
        ("num_triggers", 0),
        ("eval_num_triggers", 0),
        ("with_left_trigger", False),
        ("with_all_one_trigger_comb", False),
        ("directed_trigger_graph", False),
        ("adj_sim_head", 1),
        ("adj_sim_agg", "mean"),
        ("adj_sim_split_head", False),
        # for multi-step triggering
        ("num_triggering_steps", 1),
        # structures
        ("use_shared_dropout_proj", False),
        ("use_layer_norm_b4_biaffine", False),
        ("remove_mention_type_layer_norm", False),
        ("use_token_drop", False),
        ("guessing_decode", False),
        ("max_clique_decode", False),
        ("try_to_make_up", False),  # data building
        ("self_loop", False),  # combination decoding
        ("incremental_min_conn", -1),
        ("use_span_self_att", False),
        ("use_smooth_span_self_att", False),
        ("ment_feature_type", "plus"),
        ("ment_type_hidden_size", 32),
        ("num_mention_lstm_layer", 1),
        ("gat_alpha", 0.2),
        ("gat_num_heads", 4),
        ("gat_num_layers", 2),
        ("role_by_encoding", False),
        ("use_mention_lstm", False),
        ("mlp_before_adj_measure", False),
        ("use_field_cls_mlp", False),
        ("build_dense_connected_doc_graph", False),
        ("stop_gradient", False),
    ]

    def __init__(self, **kwargs):
        super(DEETaskSetting, self).__init__(
            self.base_key_attrs, self.base_attr_default_pairs, **kwargs
        )
        if self.run_mode == "full":
            self.train_file_name = "typed_train.json"
            self.dev_file_name = "typed_dev.json"
            self.test_file_name = "typed_test.json"
            self.doc_lang = "zh"
        elif self.run_mode == "half":
            self.train_file_name = "typed_train_1o2.json"
            self.dev_file_name = "typed_dev.json"
            self.test_file_name = "typed_test.json"
            self.doc_lang = "zh"
        elif self.run_mode == "quarter":
            self.train_file_name = "typed_train_1o4.json"
            self.dev_file_name = "typed_dev.json"
            self.test_file_name = "typed_test.json"
            self.doc_lang = "zh"
        elif self.run_mode == "1o8":
            self.train_file_name = "typed_train_1o8.json"
            self.dev_file_name = "typed_dev.json"
            self.test_file_name = "typed_test.json"
            self.doc_lang = "zh"
        elif self.run_mode == "debug":
            self.train_file_name = "typed_sample_train_48.json"
            self.dev_file_name = "typed_sample_train_48.json"
            self.test_file_name = "typed_sample_train_48.json"
            self.doc_lang = "zh"
        elif self.run_mode == "dueefin_wo_tgg":
            self.train_file_name = "dueefin_train_wo_tgg.json"
            self.dev_file_name = "dueefin_dev_wo_tgg.json"
            self.test_file_name = "dueefin_dev_wo_tgg.json"
            self.inference_file_name = "dueefin_submit_wo_tgg.json"
            self.doc_lang = "zh"
        elif self.run_mode == "dueefin_w_tgg":
            self.train_file_name = "dueefin_train_w_tgg.json"
            self.dev_file_name = "dueefin_dev_w_tgg.json"
            self.test_file_name = "dueefin_dev_w_tgg.json"
            self.inference_file_name = "dueefin_submit_w_tgg.json"
            self.doc_lang = "zh"
        elif self.run_mode == "wikievents_w_tgg":
            self.train_file_name = "train.post.wTgg.json"
            self.dev_file_name = "dev.post.wTgg.json"
            self.test_file_name = "test.post.wTgg.json"
            self.inference_file_name = "test.post.wTgg.json"
            self.doc_lang = "en"
        else:
            raise ValueError(f"run_mode: {self.run_mode} is not supported")
        if isinstance(self.filtered_data_types, str):
            self.filtered_data_types = self.filtered_data_types.split(",")


class DEETask(BasePytorchTask):
    """Doc-level Event Extraction Task"""

    def __init__(
        self,
        dee_setting,
        load_train=True,
        load_dev=True,
        load_test=True,
        load_inference=False,
        parallel_decorate=True,
    ):
        super(DEETask, self).__init__(
            dee_setting, only_master_logging=dee_setting.only_master_logging
        )
        self.best_f1 = -1.0
        self.logger = logger
        self.logging("Initializing {}".format(self.__class__.__name__))

        self.tokenizer = BertTokenizerForDocEE.from_pretrained(
            self.setting.bert_model, doc_lang=self.setting.doc_lang
        )
        self.setting.vocab_size = len(self.tokenizer.vocab)

        # get event type template
        self.event_template = get_event_template(self.setting.event_type_template)

        # get entity and event label name
        self.entity_label_list = DEEExample.get_entity_label_list(self.event_template)
        self.event_type_fields_pairs = DEEExample.get_event_type_fields_pairs(
            self.event_template
        )

        # build example loader
        self.example_loader_func = DEEExampleLoader(
            self.event_template,
            self.tokenizer,
            self.setting.rearrange_sent,
            self.setting.max_sent_len,
            drop_irr_ents_flag=self.setting.drop_irr_ents,
            include_complementary_ents=self.setting.include_complementary_ents,
            filtered_data_types=self.setting.filtered_data_types,
        )

        if not self.setting.use_token_role:
            # no token role conflicts with some settings
            if self.setting.model_type != "Doc2EDAG":
                logger.warning(
                    "Model is not Doc2EDAG! Make sure you know what you are doing here."
                )
            assert self.setting.add_greedy_dec is False
            self.setting.num_entity_labels = 3  # 0: 'O', 1: 'Begin', 2: 'Inside'
        else:
            self.setting.num_entity_labels = len(self.entity_label_list)

        self.setting.tag_id2tag_name = {
            idx: name for idx, name in enumerate(self.entity_label_list)
        }
        self.setting.ent_type2id = {
            event_type: idx
            for idx, event_type in enumerate(
                [
                    x[2:]
                    for x in filter(
                        lambda x: x.startswith("B-"),
                        self.setting.tag_id2tag_name.values(),
                    )
                ]
            )
        }
        self.setting.ent_type2id["O"] = len(self.setting.ent_type2id)

        supported_models = list_models()
        if self.setting.use_bert:
            bert_config = BertConfig.from_pretrained(self.setting.bert_model)
            bert_config.model_type = self.setting.model_type
            self.setting.update_by_dict(bert_config.__dict__)  # BertConfig dictionary
            ner_model = BERTCRFNERModel(self.setting)
        elif self.setting.use_biaffine_ner:
            ner_model = LSTMBiaffineNERModel(self.setting)
        elif self.setting.use_masked_crf:
            ner_model = LSTMMaskedCRFNERModel(self.setting)
        else:
            ner_model = None

        if self.setting.model_type == "DEPPNModel":
            bert_config = BertConfig.from_pretrained(self.setting.bert_model)
            self.setting.update_by_dict(bert_config.__dict__)
            self.setting.model_type = "DEPPNModel"

        if self.setting.model_type in {"Doc2EDAG", "Doc2EDAGModel"}:
            self.model = Doc2EDAGModel(
                self.setting,
                self.event_type_fields_pairs,
                ner_model=ner_model,
            )
        elif self.setting.model_type in {"DCFEE", "DCFEEModel"}:
            self.model = DCFEEModel(
                self.setting, self.event_type_fields_pairs, ner_model=ner_model
            )
        elif self.setting.model_type in supported_models:
            model_class = getattr(dee.models, self.setting.model_type)
            self.model = model_class(
                self.setting, self.event_type_fields_pairs, ner_model=ner_model
            )
        elif self.setting.model_type + "Model" in supported_models:
            model_class = getattr(dee.models, self.setting.model_type + "Model")
            self.model = model_class(
                self.setting, self.event_type_fields_pairs, ner_model=ner_model
            )
        else:
            raise Exception("Unsupported model type {}".format(self.setting.model_type))

        all_trainable = []
        fixed = []
        for name, param in self.model.named_parameters():
            param_num = chain_prod(param.size())
            if param.requires_grad:
                logger.info(
                    "Trainable: {:20}\t{:20}\t{}".format(
                        name, str(param.size()), param_num
                    )
                )
                all_trainable.append(param_num)
            else:
                logger.info(
                    "Untrainable: {:20}\t{:20}\t{}".format(
                        name, str(param.size()), param_num
                    )
                )
                fixed.append(param_num)

        logger.info(f"#Total Trainable Parameters: {sum(all_trainable)}")
        logger.info(f"#Total Fixed Parameters: {sum(fixed)}")

        self._decorate_model(parallel_decorate=parallel_decorate)

        # prepare optimizer
        if self.setting.use_bert:
            param_groups = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "bert" in n.lower()
                    ],
                    "lr": 3e-5,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "bert" not in n.lower()
                    ],
                    "lr": self.setting.learning_rate,
                },
            ]
            self.optimizer = optim.AdamW(param_groups)
        else:
            if self.setting.optimizer == "adamw":
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=self.setting.learning_rate
                )
            elif self.setting.optimizer == "sgd":
                self.optimizer = optim.SGD(
                    self.model.parameters(), lr=self.setting.learning_rate, momentum=0.9
                )
            else:
                self.optimizer = optim.Adam(
                    self.model.parameters(), lr=self.setting.learning_rate
                )

        # for DE-PPN
        # logic in https://github.com/HangYang-NLP/DE-PPN/blob/812cc8ba92a88049c36978e3abca7f8816c31ead/DEE/DEE_task.py#L162-L163
        # fork from https://github.com/HangYang-NLP/DE-PPN/blob/812cc8ba92a88049c36978e3abca7f8816c31ead/DEE/base_task.py#L375
        if self.setting.model_type == "DEPPNModel":
            # Prepare optimizer
            if self.setting.fp16:
                model_named_parameters = [
                    (n, param.clone().detach().to("cpu").float().requires_grad_())
                    for n, param in self.model.named_parameters()
                ]
            elif self.setting.optimize_on_cpu:
                model_named_parameters = [
                    (n, param.clone().detach().to("cpu").requires_grad_())
                    for n, param in self.model.named_parameters()
                ]
            else:
                model_named_parameters = list(self.model.named_parameters())

            no_decay = ["bias", "gamma", "beta"]
            component = ["encoder", "decoder"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in model_named_parameters
                        if n not in no_decay and component[1] not in n
                    ],
                    "weight_decay_rate": 0.01,
                    "lr": self.setting.learning_rate,
                },
                {
                    "params": [
                        p
                        for n, p in model_named_parameters
                        if n in no_decay and component[1] not in n
                    ],
                    "weight_decay_rate": 0.0,
                    "lr": self.setting.learning_rate,
                },
                {
                    "params": [
                        p
                        for n, p in model_named_parameters
                        if n not in no_decay and component[1] in n
                    ],
                    "weight_decay_rate": 0.01,
                    "lr": self.setting.deppn_decoder_lr,
                },
                {
                    "params": [
                        p
                        for n, p in model_named_parameters
                        if n in no_decay and component[1] in n
                    ],
                    "weight_decay_rate": 0.0,
                    "lr": self.setting.deppn_decoder_lr,
                },
            ]

            # for n, p in model_named_parameters:
            #     if 'ner_model' in n:
            #         p.requires_grad = False
            # self.model.ner_model.requires_grad = False

            # num_train_steps = int(
            #     len(self.train_examples)
            #     / self.setting.train_batch_size
            #     / self.setting.gradient_accumulation_steps
            #     * self.setting.num_train_epochs
            # )

            # optimizer = BertAdam(optimizer_grouped_parameters,
            #                      warmup=self.setting.warmup_proportion,
            #                      t_total=num_train_steps)

            self.optimizer = optim.AdamW(optimizer_grouped_parameters)
            # scheduler = get_linear_schedule_with_warmup(
            #     optimizer, self.setting.warmup_proportion * num_train_steps, num_train_steps
            # )

        if self.setting.use_lr_scheduler:
            # self.lr_scheduler = optim.lr_scheduler.StepLR(
            #     self.optimizer,
            #     step_size=self.setting.lr_scheduler_step,
            #     gamma=0.5)
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=max(1, int(0.02 * self.setting.num_train_epochs)),
                num_training_steps=self.setting.num_train_epochs,
            )

        # build feature converter
        if self.setting.model_type in [
            "Doc2EDAG",
            "Doc2EDAGModel",
            "DCFEE",
            "DCFEEModel",
            "GITModel",
        ]:
            convert_dataset_func = convert_dee_features_to_dataset
            self.feature_converter_func = DEEFeatureConverter(
                self.entity_label_list,
                self.event_template,
                self.setting.max_sent_len,
                self.setting.max_sent_num,
                self.tokenizer,
                include_cls=self.setting.use_bert,
                include_sep=self.setting.use_bert,
            )
        elif self.setting.model_type == "DEPPNModel":
            # use DEEFeature
            convert_dataset_func = convert_deppn_features_to_dataset
            self.feature_converter_func = DEPPNFeatureConverter(
                self.entity_label_list,
                self.event_template,
                self.setting.max_sent_len,
                self.setting.max_sent_num,
                self.tokenizer,
                include_cls=self.setting.use_bert,
                include_sep=self.setting.use_bert,
            )
        else:
            # use DEEArgRelFeature
            convert_dataset_func = convert_dee_arg_rel_features_to_dataset
            self.feature_converter_func = DEEArgRelFeatureConverter(
                self.entity_label_list,
                self.event_template,
                self.setting.max_sent_len,
                self.setting.max_sent_num,
                self.tokenizer,
                include_cls=False,
                include_sep=False,
                trigger_aware=self.setting.num_triggers != 0,
                num_triggers=self.setting.num_triggers,
                directed_graph=self.setting.directed_trigger_graph,
                try_to_make_up=self.setting.try_to_make_up,
            )

        # load data
        self._load_data(
            self.example_loader_func,
            self.feature_converter_func,
            convert_dataset_func,
            load_train=load_train,
            load_dev=load_dev,
            load_test=load_test,
            load_inference=load_inference,
        )
        # customized mini-batch producer
        self.custom_collate_fn = prepare_doc_batch_dict

        # # resume option
        # if resume_model or resume_optimizer:
        #     self.resume_checkpoint(resume_model=resume_model, resume_optimizer=resume_optimizer)

        self.min_teacher_prob = None
        self.teacher_norm = None
        self.teacher_cnt = None
        self.teacher_base = None
        self.reset_teacher_prob()

        self.logging("Successfully initialize {}".format(self.__class__.__name__))

    def reset_teacher_prob(self):
        self.min_teacher_prob = self.setting.min_teacher_prob
        if self.train_dataset is None:
            # avoid crashing when not loading training data
            num_step_per_epoch = 500
        else:
            num_step_per_epoch = int(
                len(self.train_dataset) / self.setting.train_batch_size
            )
        self.teacher_norm = num_step_per_epoch * self.setting.schedule_epoch_length
        self.teacher_base = num_step_per_epoch * self.setting.schedule_epoch_start
        self.teacher_cnt = 0

    def get_teacher_prob(self, batch_inc_flag=True):
        if self.teacher_cnt < self.teacher_base:
            prob = 1
        else:
            prob = max(
                self.min_teacher_prob,
                (self.teacher_norm - self.teacher_cnt + self.teacher_base)
                / self.teacher_norm,
            )

        if batch_inc_flag:
            self.teacher_cnt += 1

        return prob

    def get_event_idx2entity_idx2field_idx(self):
        entity_idx2entity_type = {}
        for entity_idx, entity_label in enumerate(self.entity_label_list):
            if entity_label == "O":
                entity_type = entity_label
            else:
                entity_type = entity_label[2:]

            entity_idx2entity_type[entity_idx] = entity_type

        event_idx2entity_idx2field_idx = {}
        for event_idx, (event_name, field_types, _, _) in enumerate(
            self.event_type_fields_pairs
        ):
            field_type2field_idx = {}
            for field_idx, field_type in enumerate(field_types):
                field_type2field_idx[field_type] = field_idx

            entity_idx2field_idx = {}
            for entity_idx, entity_type in entity_idx2entity_type.items():
                if entity_type in field_type2field_idx:
                    entity_idx2field_idx[entity_idx] = field_type2field_idx[entity_type]
                else:
                    entity_idx2field_idx[entity_idx] = None

            event_idx2entity_idx2field_idx[event_idx] = entity_idx2field_idx

        return event_idx2entity_idx2field_idx

    def get_loss_on_batch(self, doc_batch_dict, features=None):
        if features is None:
            features = self.train_features

        # teacher_prob = 1
        # if use_gold_span, gold spans will be used every time
        # else, teacher_prob will ensure the proportion of using gold spans
        if self.setting.use_scheduled_sampling:
            use_gold_span = False
            teacher_prob = self.get_teacher_prob()
        else:
            use_gold_span = True
            teacher_prob = 1

        try:
            loss = self.model(
                doc_batch_dict,
                features,
                use_gold_span=use_gold_span,
                train_flag=True,
                teacher_prob=teacher_prob,
            )
        except Exception:
            # DONE(tzhu): fix this issue for multi-gpu DDP training
            logger.info("-" * 30)
            logger.info(
                "Exception occurs when processing "
                + ",".join(
                    [features[ex_idx].guid for ex_idx in doc_batch_dict["ex_idx"]]
                )
            )
            raise Exception("Cannot get the loss")

        return loss

    def get_event_decode_result_on_batch(
        self, doc_batch_dict, features=None, use_gold_span=False, heuristic_type=None
    ):
        if features is None:
            raise Exception("Features mush be provided")

        if heuristic_type is None:
            event_idx2entity_idx2field_idx = None
        else:
            # this mapping is used to get span candidates for each event field
            event_idx2entity_idx2field_idx = self.get_event_idx2entity_idx2field_idx()

        batch_eval_results = self.model(
            doc_batch_dict,
            features,
            use_gold_span=use_gold_span,
            train_flag=False,
            event_idx2entity_idx2field_idx=event_idx2entity_idx2field_idx,
            heuristic_type=heuristic_type,
        )

        return batch_eval_results

    def train(self, save_cpt_flag=True, resume_base_epoch=None):
        self.logging("=" * 20 + "Start Training" + "=" * 20)
        self.reset_teacher_prob()

        # resume_base_epoch arguments have higher priority over settings
        if resume_base_epoch is None:
            # whether to resume latest cpt when restarting, very useful for preemptive scheduling clusters
            if self.setting.resume_latest_cpt:
                resume_base_epoch = self.get_latest_cpt_epoch()
            else:
                resume_base_epoch = 0

        # resume cpt if possible
        if resume_base_epoch > 0:
            self.logging("Training starts from epoch {}".format(resume_base_epoch))
            for _ in range(resume_base_epoch):
                self.get_teacher_prob()
            self.resume_cpt_at(
                resume_base_epoch, resume_model=True, resume_optimizer=True
            )
        else:
            self.logging("Training starts from scratch")

        self.base_train(
            DEETask.get_loss_on_batch,
            kwargs_dict1={},
            epoch_eval_func=DEETask.resume_save_eval_at,
            kwargs_dict2={
                "save_cpt_flag": save_cpt_flag,
                "resume_cpt_flag": False,
            },
            base_epoch_idx=resume_base_epoch,
        )
        if self.summary_writer is not None:
            self.summary_writer.close()

    def remove_cpt_before(self, epoch):
        prev_epochs = []
        for fn in os.listdir(self.setting.model_dir):
            if fn.startswith("{}.cpt".format(self.setting.cpt_file_name)):
                try:
                    ep = int(fn.split(".")[-1])
                    if ep < epoch:
                        prev_epochs.append(ep)
                except Exception:
                    continue
        for ep in prev_epochs:
            cpt_filename = "{}.cpt.{}".format(self.setting.cpt_file_name, ep)
            prev_cpt_filepath = os.path.join(self.setting.model_dir, cpt_filename)
            os.remove(prev_cpt_filepath)

    def resume_save_eval_at(self, epoch, resume_cpt_flag=False, save_cpt_flag=True):
        # if self.is_master_node():
        #     print('\nPROGRESS: {:.2f}%\n'.format(epoch / self.setting.num_train_epochs * 100))
        self.logging(
            "Current teacher prob {}".format(
                self.get_teacher_prob(batch_inc_flag=False)
            )
        )

        if resume_cpt_flag:
            self.resume_cpt_at(epoch)

        if self.is_master_node() and save_cpt_flag:
            self.save_cpt_at(epoch)
            if self.setting.remove_last_cpt:
                self.remove_cpt_before(epoch)

        if self.setting.model_type == "DCFEE":
            eval_tasks = product(["dev", "test"], [False, True], ["DCFEE-O", "DCFEE-M"])
        else:
            if self.setting.add_greedy_dec:
                eval_tasks = product(
                    ["dev", "test"], [False, True], ["GreedyDec", None]
                )
            else:
                eval_tasks = product(["dev", "test"], [False, True], [None])

        # all_id_map = defaultdict(dict)
        for task_idx, (data_type, gold_span_flag, heuristic_type) in enumerate(
            eval_tasks
        ):
            if (
                self.in_distributed_mode()
                and task_idx % dist.get_world_size() != dist.get_rank()
            ):
                continue

            if gold_span_flag:
                span_str = "gold_span"
            else:
                span_str = "pred_span"

            if heuristic_type is None:
                # store user-provided name
                model_str = self.setting.cpt_file_name.replace(".", "~")
            else:
                model_str = heuristic_type

            if data_type == "test":
                features = copy.deepcopy(self.test_features)
                dataset = copy.deepcopy(self.test_dataset)
            elif data_type == "dev":
                features = copy.deepcopy(self.dev_features)
                dataset = copy.deepcopy(self.dev_dataset)
            else:
                raise Exception("Unsupported data type {}".format(data_type))
            decode_dump_name = decode_dump_template.format(
                data_type, span_str, model_str, epoch
            )
            eval_dump_name = eval_dump_template.format(
                data_type, span_str, model_str, epoch
            )
            _, measures = self.eval(
                features,
                dataset,
                use_gold_span=gold_span_flag,
                heuristic_type=heuristic_type,
                dump_decode_pkl_name=decode_dump_name,
                dump_eval_json_name=eval_dump_name,
            )
            if self.is_master_node() and data_type == "dev" and gold_span_flag is False:
                curr_f1 = measures["overall"]["overall"]["MicroF1"]
                self.logging(
                    f"Epoch: {epoch}, Current F1: {curr_f1 * 100:.3f}, Best F1: {self.best_f1 * 100:.3f}, is the best: {curr_f1 > self.best_f1}"
                )
                if curr_f1 > self.best_f1:
                    self.best_f1 = curr_f1
                    if self.setting.save_best_cpt:
                        self.save_cpt_at(epoch)
                        if self.setting.remove_last_cpt:
                            self.remove_cpt_before(epoch)

    def save_cpt_at(self, epoch):
        self.save_checkpoint(
            cpt_file_name="{}.cpt.{}".format(self.setting.cpt_file_name, epoch),
            epoch=epoch,
        )

    def resume_cpt_at(self, epoch, resume_model=True, resume_optimizer=False):
        self.resume_checkpoint(
            cpt_file_name="{}.cpt.{}".format(self.setting.cpt_file_name, epoch),
            resume_model=resume_model,
            resume_optimizer=resume_optimizer,
        )

    def get_latest_cpt_epoch(self):
        prev_epochs = []
        for fn in os.listdir(self.setting.model_dir):
            if fn.startswith("{}.cpt".format(self.setting.cpt_file_name)):
                try:
                    epoch = int(fn.split(".")[-1])
                    prev_epochs.append(epoch)
                except Exception:
                    continue
        prev_epochs.sort()

        if len(prev_epochs) > 0:
            latest_epoch = prev_epochs[-1]
            self.logging(
                "Pick latest epoch {} from {}".format(latest_epoch, str(prev_epochs))
            )
        else:
            latest_epoch = 0
            self.logging("No previous epoch checkpoints, just start from scratch")

        return latest_epoch

    def eval(
        self,
        features,
        dataset,
        use_gold_span=False,
        heuristic_type=None,
        dump_decode_pkl_name=None,
        dump_eval_json_name=None,
    ):
        self.logging("=" * 20 + "Start Evaluation" + "=" * 20)

        if dump_decode_pkl_name is not None:
            dump_decode_pkl_path = os.path.join(
                self.setting.output_dir, dump_decode_pkl_name
            )
            self.logging("Dumping decode results into {}".format(dump_decode_pkl_name))
        else:
            dump_decode_pkl_path = None

        total_event_decode_results = self.base_eval(
            dataset,
            DEETask.get_event_decode_result_on_batch,
            reduce_info_type="none",
            dump_pkl_path=dump_decode_pkl_path,
            features=features,
            use_gold_span=use_gold_span,
            heuristic_type=heuristic_type,
        )

        self.logging("Measure DEE Prediction")

        if dump_eval_json_name is not None:
            dump_eval_json_path = os.path.join(
                self.setting.output_dir, dump_eval_json_name
            )
            self.logging("Dumping eval results into {}".format(dump_eval_json_path))
        else:
            dump_eval_json_path = None

        total_eval_res = measure_dee_prediction(
            self.event_type_fields_pairs,
            features,
            total_event_decode_results,
            self.setting.event_relevant_combination,
            dump_json_path=dump_eval_json_path,
        )
        if self.is_master_node():
            dataset_name = span_type = "unknown"
            epoch = 0
            if dump_eval_json_name is not None:
                dataset_name, span_type = dump_eval_json_name.split(".")[1:3]
                epoch = dump_eval_json_name.split(".")[-2]
            elif dump_decode_pkl_name is not None:
                dataset_name, span_type = dump_decode_pkl_name.split(".")[1:3]
                epoch = dump_eval_json_name.split(".")[-2]
            epoch = int(epoch)
            measure_list = [
                "classification",
                "entity",
                "combination",
                "overall",
                "instance",
            ]
            if self.summary_writer is not None:
                for measure_name in measure_list:
                    self.summary_writer.add_scalars(
                        f"{dataset_name}/{span_type}/{measure_name}",
                        {
                            "o2o": total_eval_res["o2o"][measure_name]["MicroF1"],
                            "o2m": total_eval_res["o2m"][measure_name]["MicroF1"],
                            "m2m": total_eval_res["m2m"][measure_name]["MicroF1"],
                            "overall": total_eval_res["overall"][measure_name][
                                "MicroF1"
                            ],
                        },
                        global_step=epoch,
                    )

                adj_mat_measures = dict()
                raw_combination_measures = dict()
                connection_measures = dict()
                trigger_measures = dict()
                for doc_type in ["o2o", "o2m", "m2m", "overall"]:
                    if "adj_mat" in total_eval_res[doc_type]:
                        adj_mat_measures.update(
                            {doc_type: total_eval_res[doc_type]["adj_mat"]["Accuracy"]}
                        )
                    if "rawCombination" in total_eval_res[doc_type]:
                        raw_combination_measures.update(
                            {
                                doc_type: total_eval_res[doc_type]["rawCombination"][
                                    "MicroF1"
                                ]
                            }
                        )
                    if "connection" in total_eval_res[doc_type]:
                        connection_measures.update(
                            {
                                doc_type: total_eval_res[doc_type]["connection"][
                                    "MicroF1"
                                ]
                            }
                        )
                    if "trigger" in total_eval_res[doc_type]:
                        trigger_measures.update(
                            {doc_type: total_eval_res[doc_type]["trigger"]["MicroF1"]}
                        )

                if adj_mat_measures:
                    self.summary_writer.add_scalars(
                        f"{dataset_name}/{span_type}/adj_mat",
                        adj_mat_measures,
                        global_step=epoch,
                    )
                if raw_combination_measures:
                    self.summary_writer.add_scalars(
                        f"{dataset_name}/{span_type}/rawCombination",
                        raw_combination_measures,
                        global_step=epoch,
                    )
                if connection_measures:
                    self.summary_writer.add_scalars(
                        f"{dataset_name}/{span_type}/connection",
                        connection_measures,
                        global_step=epoch,
                    )
                if trigger_measures:
                    self.summary_writer.add_scalars(
                        f"{dataset_name}/{span_type}/trigger",
                        trigger_measures,
                        global_step=epoch,
                    )

        return total_event_decode_results, total_eval_res

    def reevaluate_dee_prediction(
        self,
        max_epoch=100,
        target_file_pre="dee_eval",
        target_file_suffix=".pkl",
        dump_flag=False,
    ):
        """Enumerate the evaluation directory to collect all dumped evaluation results"""
        eval_dir_path = self.setting.output_dir
        logger.info("Re-evaluate dee predictions from {}".format(eval_dir_path))
        doc_type2data_span_type2model_str2epoch_res_list = {}
        pkl_match_name = os.path.join(
            eval_dir_path, f"{target_file_pre}.*{target_file_suffix}"
        )
        pkl_matched_names = glob.glob(pkl_match_name)
        pbar = tqdm(pkl_matched_names, desc="ReEval", ncols=80, ascii=True)
        for fn in pbar:
            fn = os.path.split(fn)[-1]
            fn_splits = fn.split(".")
            if (
                fn.startswith(target_file_pre)
                and fn.endswith(target_file_suffix)
                and len(fn_splits) == 6
            ):
                _, data_type, span_type, model_str, epoch, _ = fn_splits
                epoch = int(epoch)
                if epoch > max_epoch:
                    continue

                if data_type == "dev":
                    features = self.dev_features
                elif data_type == "test":
                    features = self.test_features
                else:
                    raise Exception("Unsupported data type {}".format(data_type))

                fp = os.path.join(eval_dir_path, fn)
                # self.logging('Re-evaluating {}'.format(fp))
                event_decode_results = default_load_pkl(fp)
                total_eval_res = measure_dee_prediction(
                    self.event_template.event_type_fields_list,
                    features,
                    event_decode_results,
                    self.setting.event_relevant_combination,
                )

                for doc_type in ["o2o", "o2m", "m2m", "overall"]:
                    if doc_type not in doc_type2data_span_type2model_str2epoch_res_list:
                        doc_type2data_span_type2model_str2epoch_res_list[doc_type] = {}

                    data_span_type = (data_type, span_type)
                    if (
                        data_span_type
                        not in doc_type2data_span_type2model_str2epoch_res_list[
                            doc_type
                        ]
                    ):
                        doc_type2data_span_type2model_str2epoch_res_list[doc_type][
                            data_span_type
                        ] = {}
                    model_str2epoch_res_list = (
                        doc_type2data_span_type2model_str2epoch_res_list[doc_type][
                            data_span_type
                        ]
                    )

                    if model_str not in model_str2epoch_res_list:
                        model_str2epoch_res_list[model_str] = []
                    epoch_res_list = model_str2epoch_res_list[model_str]

                    epoch_res_list.append((epoch, total_eval_res[doc_type]))

                if dump_flag:
                    fp = fp.rstrip(".pkl") + ".json"
                    # self.logging('Dumping {}'.format(fp))
                    default_dump_json(total_eval_res, fp)
        logger.info(pbar)

        for (
            doc_type,
            data_span_type2model_str2epoch_res_list,
        ) in doc_type2data_span_type2model_str2epoch_res_list.items():
            for (
                data_span_type,
                model_str2epoch_res_list,
            ) in data_span_type2model_str2epoch_res_list.items():
                for model_str, epoch_res_list in model_str2epoch_res_list.items():
                    epoch_res_list.sort(key=lambda x: x[0])

        return doc_type2data_span_type2model_str2epoch_res_list

    def ensemble_dee_prediction(self, curr_best_pkl_filepath, esmb_best_pkl_filepath):
        """ensembling based on absent-filling strategy"""
        curr_encode_results = default_load_pkl(curr_best_pkl_filepath)
        esmb_encode_results = default_load_pkl(esmb_best_pkl_filepath)
        new_results = []
        for curr, esmb in zip(curr_encode_results, esmb_encode_results):
            if all(x is None for x in curr[2]):
                new_results.append(esmb)
            else:
                new_results.append(curr)

        total_eval_res = measure_dee_prediction(
            self.event_template.event_type_fields_list,
            self.test_features,
            new_results,
            self.setting.event_relevant_combination,
        )
        print_data = []
        results = {
            "ModelType": "ensemble",
            "o2o": {
                "classification": {"precision": None, "recall": None, "f1": None},
                "entity": {"precision": None, "recall": None, "f1": None},
                "combination": {"precision": None, "recall": None, "f1": None},
                "rawCombination": {"precision": None, "recall": None, "f1": None},
                "overall": {"precision": None, "recall": None, "f1": None},
                "instance": {"precision": None, "recall": None, "f1": None},
            },
            "o2m": {
                "classification": {"precision": None, "recall": None, "f1": None},
                "entity": {"precision": None, "recall": None, "f1": None},
                "combination": {"precision": None, "recall": None, "f1": None},
                "rawCombination": {"precision": None, "recall": None, "f1": None},
                "overall": {"precision": None, "recall": None, "f1": None},
                "instance": {"precision": None, "recall": None, "f1": None},
            },
            "m2m": {
                "classification": {"precision": None, "recall": None, "f1": None},
                "entity": {"precision": None, "recall": None, "f1": None},
                "combination": {"precision": None, "recall": None, "f1": None},
                "rawCombination": {"precision": None, "recall": None, "f1": None},
                "overall": {"precision": None, "recall": None, "f1": None},
                "instance": {"precision": None, "recall": None, "f1": None},
            },
            "overall": {
                "classification": {"precision": None, "recall": None, "f1": None},
                "entity": {"precision": None, "recall": None, "f1": None},
                "combination": {"precision": None, "recall": None, "f1": None},
                "rawCombination": {"precision": None, "recall": None, "f1": None},
                "overall": {"precision": None, "recall": None, "f1": None},
                "instance": {"precision": None, "recall": None, "f1": None},
            },
        }
        MEASURE_TYPES = [
            "classification",
            "entity",
            "combination",
            "rawCombination",
            "overall",
            "instance",
        ]
        headers = []
        for measure_type in MEASURE_TYPES:
            if measure_type in total_eval_res["overall"].keys():
                headers.append(measure_type)
        header = "Data\t{}".format(
            "\t".join(list(map(lambda x: "{:20}".format(x.title()), headers)))
        )
        logger.info(header)
        logger.info("    \t{}".format("Prec\tRecall\tF1\t" * len(headers)))
        for data_type in ["o2o", "o2m", "m2m", "overall"]:
            result = total_eval_res[data_type]
            tmp_print = [data_type]
            for measure_type in MEASURE_TYPES:
                if measure_type in result:
                    tmp_print.extend(
                        [
                            result[measure_type]["MicroPrecision"],
                            result[measure_type]["MicroRecall"],
                            result[measure_type]["MicroF1"],
                        ]
                    )
                    results[data_type][measure_type]["precision"] = "{:.3f}".format(
                        result[measure_type]["MicroPrecision"] * 100
                    )
                    results[data_type][measure_type]["recall"] = "{:.3f}".format(
                        result[measure_type]["MicroRecall"] * 100
                    )
                    results[data_type][measure_type]["f1"] = "{:.3f}".format(
                        result[measure_type]["MicroF1"] * 100
                    )
            print_data.append(tmp_print)

        for ds in print_data:
            for d in ds:
                if isinstance(d, float):
                    logger.info("{:.3f}".format(d * 100), end="\t")
                else:
                    logger.info("{}".format(d), end="\t")
            logger.info()

    @torch.no_grad()
    def predict_one(self, sents: Union[str, List[str]]):
        """
        sents:
            List of sentences of *one* doc, or one string for one doc with
            the string being segmented into sentences automatically by default.
        """
        self.model.eval()
        guid = "PREDICTION"
        data = convert_string_to_raw_input(guid, sents)
        examples = [
            self.example_loader_func.convert_dict_to_example(
                data[0], data[1], only_inference=True
            )
        ]
        features = self.feature_converter_func(examples)
        batch = self.custom_collate_fn(features)
        batch = self.set_batch_to_device(batch)
        batch_info = self.get_event_decode_result_on_batch(
            batch, features=features, use_gold_span=False, heuristic_type=None
        )
        example = examples[0]
        doc_fea = features[0]
        result = batch_info[0]

        doc_id = doc_fea.guid
        event_list = []
        mspans = []
        event_types = []
        for eid, r in enumerate(result[1]):
            if r == 1:
                event_types.append(self.event_type_fields_pairs[eid][0])

        doc_arg_rel_info = result[3]
        mention_drange_list = doc_arg_rel_info.mention_drange_list
        mention_type_list = doc_arg_rel_info.mention_type_list
        doc_token_ids = doc_fea.doc_token_ids.detach().tolist()
        for drange, ment_type in zip(mention_drange_list, mention_type_list):
            mspan = self.tokenizer.convert_ids_to_tokens(
                doc_token_ids[drange[0]][drange[1] : drange[2]]
            )
            if all(x.upper() != "[UNK]" for x in mspan):
                mspan = "".join(mspan)
                offset = int(self.feature_converter_func.include_cls)
                matched_drange = [drange[0], drange[1] - offset, drange[2] - offset]
            else:
                mspan, matched_drange = match_arg(
                    example.sentences,
                    doc_fea.doc_token_ids.numpy(),
                    doc_token_ids[drange[0]][drange[1] : drange[2]],
                    offset=int(self.feature_converter_func.include_cls),
                )
            mtype = self.setting.tag_id2tag_name[ment_type][2:]
            t_mspan = {"mspan": mspan, "mtype": mtype, "drange": matched_drange}
            if t_mspan not in mspans:
                mspans.append(t_mspan)

        for event_idx, events in enumerate(result[2]):
            if events is None:
                continue
            for ins in events:
                if all(x is None for x in ins):
                    continue
                tmp_ins = {
                    "event_type": self.event_template.event_type_fields_list[event_idx][
                        0
                    ],
                    "arguments": None,
                }
                arguments = []
                for field_idx, args in enumerate(ins):
                    if args is None:
                        continue
                    if not isinstance(args, set):
                        args = {args}
                    for arg in args:
                        arg_tmp = self.tokenizer.convert_ids_to_tokens(arg)
                        if all(x.upper() != "[UNK]" for x in arg_tmp):
                            real_arg = "".join(arg_tmp)
                        else:
                            real_arg, _ = match_arg(
                                example.sentences,
                                doc_fea.doc_token_ids.numpy(),
                                arg,
                                offset=int(self.feature_converter_func.include_cls),
                            )
                            if real_arg is None:
                                self.logging(
                                    f"doc: {doc_id}, arg ({arg_tmp}) with UNK but original text not found",
                                    level=logging.WARNING,
                                )
                                real_arg = arg_tmp
                        arguments.append(
                            {
                                "role": self.event_template.event_type_fields_list[
                                    event_idx
                                ][1][field_idx],
                                "argument": real_arg,
                            }
                        )
                tmp_ins["arguments"] = arguments
                if tmp_ins not in event_list:
                    event_list.append(tmp_ins)

        event_list_merge_flag = [True for _ in range(len(event_list))]
        for ins1, ins2 in combinations(enumerate(event_list), 2):
            if ins1[1]["event_type"] == ins2[1]["event_type"]:
                ins1_args = {
                    (arg["role"], arg["argument"]) for arg in ins1[1]["arguments"]
                }
                ins2_args = {
                    (arg["role"], arg["argument"]) for arg in ins2[1]["arguments"]
                }
                if ins1_args == ins2_args or ins2_args.issubset(ins1_args):
                    event_list_merge_flag[ins2[0]] = False
                elif ins1_args.issubset(ins2_args):
                    event_list_merge_flag[ins1[0]] = False
        new_event_list = []
        for flag, events in zip(event_list_merge_flag, event_list):
            if flag:
                new_event_list.append(events)

        doc_res = {
            "id": doc_id,
            "event_list": new_event_list,
            "comments": {
                "pred_types": event_types,
                "mspans": mspans,
                "sentences": example.sentences,
            },
        }
        return doc_res

    def debug_display(self, doc_type, span_type, epoch, midout_dir):
        import json

        from transformers import BertTokenizer

        from dee.helper import DEEArgRelFeature, DEEFeature, measure_event_table_filling
        from dee.utils import (
            convert_role_fea_event_obj_to_standard,
            extract_combinations_from_event_objs,
            fill_diag,
            recover_ins,
        )

        tokenizer = BertTokenizer.from_pretrained(self.setting.bert_model)

        features = self.test_features
        output_dir = os.path.join(midout_dir, f"{doc_type}/{span_type}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fn = f"dee_eval.test.{span_type}.{self.setting.model_type}.{epoch}.pkl"
        fp = os.path.join(self.setting.output_dir, fn)
        # self.logging('Re-evaluating {}'.format(fp))
        event_decode_results = default_load_pkl(fp)

        is_cg = False
        if len(features) > 0:
            if all(isinstance(feature, DEEArgRelFeature) for feature in features):
                is_cg = True
            elif all(isinstance(feature, DEEFeature) for feature in features):
                is_cg = False
            else:
                raise ValueError("Not all the features are in the same type!")

        new_event_decode_results = copy.deepcopy(event_decode_results)
        filtered_event_decode_results = []
        for doc_fea, decode_result in zip(features, new_event_decode_results):
            if (
                doc_type != "overall"
                and doc_fea.doc_type != {"o2o": 0, "o2m": 1, "m2m": 2}[doc_type]
            ):
                continue
            filtered_event_decode_results.append(decode_result)

        pred_record_mat_list = []
        gold_record_mat_list = []
        pred_event_types = []
        gold_event_types = []
        pred_spans_token_tuple_list = []
        gold_spans_token_tuple_list = []
        pred_adj_mats = []
        gold_adj_mats = []
        pred_combinations = []
        gold_combinations = []
        new_features = []
        for term in filtered_event_decode_results:
            ex_idx, pred_event_type_labels, pred_record_mat, doc_span_info = term[:4]
            doc_fea = features[ex_idx]
            new_features.append(doc_fea)

            if is_cg:
                pred_adj_mat, event_idx2combinations = term[4:6]
                pred_adj_mats.append(pred_adj_mat)
                gold_adj_mats.append([doc_fea.whole_arg_rel_mat.reveal_adj_mat()])
                tmp_pred_combinations = set()
                for combinations_ in event_idx2combinations:
                    combinations_ = [
                        tuple(
                            sorted(
                                [doc_span_info.span_token_tup_list[arg] for arg in comb]
                            )
                        )
                        for comb in combinations_
                    ]
                    tmp_pred_combinations.update(set(combinations_))
                pred_combinations.append(tmp_pred_combinations)
                # convert doc_fea.event_arg_idxs_objs_list and remove the role labels
                doc_fea.event_arg_idxs_objs_list = remove_event_obj_roles(
                    doc_fea.event_arg_idxs_objs_list, self.event_type_fields_pairs
                )
                tmp_gold_combinations = extract_combinations_from_event_objs(
                    doc_fea.event_arg_idxs_objs_list
                )
                tmp_gold_combinations = set(
                    [
                        tuple(
                            sorted([doc_fea.span_token_ids_list[arg] for arg in comb])
                        )
                        for comb in tmp_gold_combinations
                    ]
                )
                gold_combinations.append(tmp_gold_combinations)

            pred_event_types.append(pred_event_type_labels)
            gold_event_types.append(doc_fea.event_type_labels)
            pred_spans_token_tuple_list.append(doc_span_info.span_token_tup_list)
            gold_spans_token_tuple_list.append(doc_fea.span_token_ids_list)

            pred_record_mat = [
                [
                    [
                        tuple(arg_tup) if arg_tup is not None else None
                        for arg_tup in pred_record
                    ]
                    for pred_record in pred_records
                ]
                if pred_records is not None
                else None
                for pred_records in pred_record_mat
            ]
            gold_record_mat = [
                [
                    [
                        tuple(doc_fea.span_token_ids_list[arg_idx])
                        if arg_idx is not None
                        else None
                        for arg_idx in event_arg_idxs
                    ]
                    for event_arg_idxs in event_arg_idxs_objs
                ]
                if event_arg_idxs_objs is not None
                else None  # for events in each event type
                for event_arg_idxs_objs in doc_fea.event_arg_idxs_objs_list
            ]
            pred_record_mat_list.append(pred_record_mat)
            gold_record_mat_list.append(gold_record_mat)

        g_eval_res = measure_event_table_filling(
            pred_record_mat_list,
            gold_record_mat_list,
            self.event_template.event_type_fields_list,
            pred_event_types,
            gold_event_types,
            pred_spans_token_tuple_list,
            gold_spans_token_tuple_list,
            pred_adj_mats=pred_adj_mats,
            gold_adj_mats=gold_adj_mats,
            pred_combinations=pred_combinations,
            gold_combinations=gold_combinations,
            dict_return=True,
        )
        print_data = {
            "classification": {
                "p": g_eval_res["classification"]["MicroPrecision"],
                "r": g_eval_res["classification"]["MicroRecall"],
                "f1": g_eval_res["classification"]["MicroF1"],
            },
            "entity": {
                "p": g_eval_res["entity"]["MicroPrecision"],
                "r": g_eval_res["entity"]["MicroRecall"],
                "f1": g_eval_res["entity"]["MicroF1"],
            },
            "combination": {
                "p": g_eval_res["combination"]["MicroPrecision"],
                "r": g_eval_res["combination"]["MicroRecall"],
                "f1": g_eval_res["combination"]["MicroF1"],
            },
            "rawCombination": {
                "p": g_eval_res["rawCombination"]["MicroPrecision"],
                "r": g_eval_res["rawCombination"]["MicroRecall"],
                "f1": g_eval_res["rawCombination"]["MicroF1"],
            },
            "overall": {
                "p": g_eval_res["overall"]["MicroPrecision"],
                "r": g_eval_res["overall"]["MicroRecall"],
                "f1": g_eval_res["overall"]["MicroF1"],
            },
            "instance": {
                "p": g_eval_res["instance"]["MicroPrecision"],
                "r": g_eval_res["instance"]["MicroRecall"],
                "f1": g_eval_res["instance"]["MicroF1"],
            },
            "adj_mat": g_eval_res["adj_mat"]["Accuracy"],
        }
        logger.info(json.dumps(print_data, indent=2, ensure_ascii=False))
        type_names = [x[0] for x in self.event_template.event_type_fields_list]
        for (
            doc_fea,
            pred_record_mat,
            gold_record_mat,
            pred_event_type,
            gold_event_type,
            pred_spans_token_tuple,
            gold_spans_token_tuple,
            pred_adj_mat,
            gold_adj_mat,
            pred_combination,
            gold_combination,
        ) in zip(
            new_features,
            pred_record_mat_list,
            gold_record_mat_list,
            pred_event_types,
            gold_event_types,
            pred_spans_token_tuple_list,
            gold_spans_token_tuple_list,
            pred_adj_mats,
            gold_adj_mats,
            pred_combinations,
            gold_combinations,
        ):
            if pred_record_mat != gold_record_mat:
                continue
            texts = []
            for line in doc_fea.doc_token_ids:
                texts.append(
                    "".join(
                        filter(
                            lambda x: x != "[PAD]",
                            tokenizer.convert_ids_to_tokens(line.tolist()),
                        )
                    )
                )
            mid_result = {
                "doc_type": ["o2o", "o2m", "m2m", "unk"][doc_fea.doc_type],
                "span_type": span_type,
                "guid": doc_fea.guid,
                "texts": texts,
                "pred_event_type": list(
                    map(
                        lambda x: x[0],
                        filter(lambda x: x[1] == 1, zip(type_names, pred_event_type)),
                    )
                ),
                "gold_event_type": list(
                    map(
                        lambda x: x[0],
                        filter(lambda x: x[1] == 1, zip(type_names, gold_event_type)),
                    )
                ),
                "pred_ents": [
                    "".join(tokenizer.convert_ids_to_tokens(span))
                    for span in pred_spans_token_tuple
                ],
                "gold_ents": [
                    "".join(tokenizer.convert_ids_to_tokens(span))
                    for span in gold_spans_token_tuple
                ],
                "pred_adj_mat": fill_diag(pred_adj_mat[0], -1),
                "gold_adj_mat": gold_adj_mat[0],
                "pred_comb": [
                    ["".join(tokenizer.convert_ids_to_tokens(arg)) for arg in comb]
                    for comb in pred_combination
                ],
                "gold_comb": [
                    ["".join(tokenizer.convert_ids_to_tokens(arg)) for arg in comb]
                    for comb in gold_combination
                ],
                "pred_ins": recover_ins(
                    self.event_template.event_type_fields_list,
                    tokenizer.convert_ids_to_tokens,
                    pred_record_mat,
                ),
                "gold_ins": recover_ins(
                    self.event_template.event_type_fields_list,
                    tokenizer.convert_ids_to_tokens,
                    gold_record_mat,
                ),
            }
            with open(
                os.path.join(output_dir, f"{doc_fea.guid}.json"), "wt", encoding="utf-8"
            ) as fout:
                json.dump(mid_result, fout, ensure_ascii=False, indent=2)

    def inference(self, dump_filepath=None, resume_epoch=1):
        import json

        import torch

        self.resume_cpt_at(resume_epoch)

        self.logging(
            "=" * 20 + "Start Inference, Will Dump to: " + dump_filepath + "=" * 20
        )

        # prepare data loader
        eval_dataloader = self.prepare_data_loader(
            self.inference_dataset, self.setting.eval_batch_size, rand_flag=False
        )

        # enter eval mode
        total_info = []
        if self.model is not None:
            self.model.eval()

        iter_desc = "Inference"
        if self.in_distributed_mode():
            iter_desc = "Rank {} {}".format(dist.get_rank(), iter_desc)

        for step, batch in enumerate(
            tqdm(eval_dataloader, desc=iter_desc, ncols=80, ascii=True)
        ):
            batch = self.set_batch_to_device(batch)

            with torch.no_grad():
                # this func must run batch_info = model(batch_input)
                # and metrics is an instance of torch.Tensor with Size([batch_size, ...])
                # to fit the DataParallel and DistributedParallel functionality
                batch_info = self.get_event_decode_result_on_batch(
                    batch,
                    features=self.inference_features,
                    use_gold_span=False,
                    heuristic_type=None,
                )
            # append metrics from this batch to event_info
            if isinstance(batch_info, torch.Tensor):
                total_info.append(
                    batch_info.detach().cpu()  # collect results in cpu memory
                )
            else:
                # batch_info is a list of some info on each example
                total_info.extend(batch_info)

        if isinstance(total_info[0], torch.Tensor):
            # transform event_info to torch.Tensor
            total_info = torch.cat(total_info, dim=0)

        assert (
            len(self.inference_examples)
            == len(self.inference_features)
            == len(total_info)
        )
        # example_list = self.inference_examples
        # feature_list = self.inference_features
        example_list = []
        feature_list = []
        for info in total_info:
            example_list.append(self.inference_examples[info[0]])
            feature_list.append(self.inference_features[info[0]])

        if dump_filepath is not None:
            with open(dump_filepath, "wt", encoding="utf-8") as fout:
                for example, doc_fea, result in zip(
                    example_list, feature_list, total_info
                ):
                    assert doc_fea.ex_idx == result[0]

                    doc_id = doc_fea.guid
                    event_list = []
                    mspans = []
                    event_types = []
                    for eid, r in enumerate(result[1]):
                        if r == 1:
                            event_types.append(self.event_type_fields_pairs[eid][0])

                    doc_arg_rel_info = result[3]
                    mention_drange_list = doc_arg_rel_info.mention_drange_list
                    mention_type_list = doc_arg_rel_info.mention_type_list
                    doc_token_ids = doc_fea.doc_token_ids.detach().tolist()

                    for drange, ment_type in zip(
                        mention_drange_list, mention_type_list
                    ):
                        mspan = self.tokenizer.convert_ids_to_tokens(
                            doc_token_ids[drange[0]][drange[1] : drange[2]]
                        )
                        if all(x.upper() != "[UNK]" for x in mspan):
                            mspan = "".join(mspan)
                            offset = int(self.feature_converter_func.include_cls)
                            matched_drange = [
                                drange[0],
                                drange[1] - offset,
                                drange[2] - offset,
                            ]
                        else:
                            mspan, matched_drange = match_arg(
                                example.sentences,
                                doc_fea.doc_token_ids.numpy(),
                                doc_token_ids[drange[0]][drange[1] : drange[2]],
                                offset=int(self.feature_converter_func.include_cls),
                            )
                        mtype = self.setting.tag_id2tag_name[ment_type][2:]
                        t_mspan = {
                            "mspan": mspan,
                            "mtype": mtype,
                            "drange": matched_drange,
                        }
                        if t_mspan not in mspans:
                            mspans.append(t_mspan)

                    for event_idx, events in enumerate(result[2]):
                        if events is None:
                            continue
                        for ins in events:
                            if all(x is None for x in ins):
                                continue
                            tmp_ins = {
                                "event_type": self.event_template.event_type_fields_list[
                                    event_idx
                                ][
                                    0
                                ],
                                "arguments": None,
                            }
                            arguments = []
                            for field_idx, args in enumerate(ins):
                                if args is None:
                                    continue
                                if not isinstance(args, set):
                                    args = {args}
                                for arg in args:
                                    arg_tmp = self.tokenizer.convert_ids_to_tokens(arg)
                                    if all(x.upper() != "[UNK]" for x in arg_tmp):
                                        real_arg = "".join(arg_tmp)
                                    else:
                                        real_arg, _ = match_arg(
                                            example.sentences,
                                            doc_fea.doc_token_ids.numpy(),
                                            arg,
                                            offset=int(
                                                self.feature_converter_func.include_cls
                                            ),
                                        )
                                        if real_arg is None:
                                            self.logging(
                                                f"doc: {doc_id}, arg ({arg_tmp}) with UNK but original text not found",
                                                level=logging.WARNING,
                                            )
                                            real_arg = arg_tmp
                                    arguments.append(
                                        {
                                            "role": self.event_template.event_type_fields_list[
                                                event_idx
                                            ][
                                                1
                                            ][
                                                field_idx
                                            ],
                                            "argument": real_arg,
                                        }
                                    )
                            tmp_ins["arguments"] = arguments
                            event_list.append(tmp_ins)

                    doc_res = {
                        "id": doc_id,
                        "event_list": event_list,
                        "comments": {
                            "pred_types": event_types,
                            "mspans": mspans,
                            "sentences": example.sentences,
                        },
                    }
                    fout.write(f"{json.dumps(doc_res, ensure_ascii=False)}\n")
                    fout.flush()
            self.logging(f"Results dumped to {dump_filepath}")
