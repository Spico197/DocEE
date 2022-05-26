.. toctree::
   :maxdepth: 2
   :caption: Contents:


Evaluation
==========

:Authors:
    Tong Zhu

:Last Update: Jan. 4th, 2021


You may wondering what are those terms in
``Exps/<task_name>/Output/dee_eval.(dev|test).(pred|gold)_span.<model_name>.<epoch>.json``.
Here are the explanation.

Doc Type
--------

Document types are combined with the number of event types and the number of event instances per type.

o2o
    There is only one event type with one instance.

o2m
    There are only one event type with multiple instances.

m2m
    There are multiple event types.

Metrics
-------

classification
    The event type classification measurements.

entity
    The Named Entity Recognition (NER) part of measurements.

overall
    The final metric with role-level evaluation as introduced in Doc2EDAG [#Doc2EDAG]_.

instance
    The instance-level measurements.
    One instance is recognised as True Positive (TP) iff all the argument roles have filled with correct arguments.

trigger
    For PTPCG, ``trigger`` means the evaluation of pseudo triggers.

adj_mat
    For PTPCG, ``adj_mat`` means the evaluation of adjacent matrix for each document.

connection
    For PTPCG, ``connection`` means the evaluation of connections between pseudo triggers and ordinary arguments.

rawCombination
    In PTPCG, ``rawCombination`` is the combination evaluation results
    after the BK extraction without further instance generation and argument filtering.

combination
    ``combination`` is the combination evaluation results
    after the final instance generation process.
    Some arguments in ``rawCombination`` may be filtered out.

References
----------

.. [#Doc2EDAG] Shun Zheng, Wei Cao, Wei Xu, and Jiang Bian. 2020. Doc2EDAG: An end-to-end document-level framework for Chinese financial event extraction. EMNLP-IJCNLP 2019 - 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing, Proceedings of the Conference:337–346.
