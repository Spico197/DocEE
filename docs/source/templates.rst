.. toctree::
   :maxdepth: 2
   :caption: Contents:


Templates
=========

:Authors:
    Tong Zhu

:Last Update: May 26th, 2022


Currently, DocEE supports ChFinAnn_ and DuEE-fin_ datasets with the following templates in ``dee/event_types``:

ChFinAnn
--------

ChFinAnn is a Chinese dataset for document-level financial event extraction.

zheng2019
    The original template used in Doc2EDAG_ and GIT_. No ``OtherType`` entities are included.

zheng2019_trigger_graph
    Used by ``PTPCG``, with ``OtherType`` additional entities included.
    As introduced in the original paper, PTPCG add additional entities which includes:
    - Original ``OtherType`` entities in ChFinAnn. To use this part of entities, ``OtherType`` must be included in ``common_fields`` in template files.
    - Additional entities matched by regular expressions (must set ``include_complementary_ents=True`` in settings)

zheng2019_trigger_graph_high_importance
    The same with ``zheng2019_trigger_graph``, with detailed importance scores listed.
    Importance scores are calculated to select pseudo triggers in PTPCG.
    We can select the best pseudo triggers with the highest importance scores.

zheng2019_trigger_graph_mid_importance
    Similar to ``zheng2019_trigger_graph_high_importance``, with middle importance scores.
    i.e. The pseudo trigger quality is lower than the best, resulting in performance decline.
    This is used for ablation study and analysis.

zheng2019_trigger_graph_low_importance
    Similar to ``zheng2019_trigger_graph_mid_importance`` with the lowest importance scores.

zheng2019_trigger_graph_no_OtherType
    Similar to ``zheng2019_trigger_graph``, without ``OtherType`` in ``common_fields``.


DuEE-fin
--------

dueefin_wo_tgg
    DuEE-fin template without manually annotated triggers.
    Here, ``Trigger`` is not a role in an event type schema.

dueefin_wo_tgg_mid_importance
    Similar to ``zheng2019_trigger_graph_mid_importance``, pseudo triggers are selected with middle importance scores.
    For analysis usage only.

dueefin_wo_tgg_low_importance
    Similar to ``dueefin_wo_tgg_mid_importance``, pseudo triggers are selected with the lowest importance scores.

dueefin_wo_tgg_woOtherType
    ``OtherType`` is included as default in the **above** templates.
    If you want to make ablation studies or further analyses on the effect of additional entities, you may want to use this template.

dueefin_w_tgg
    Manually annotated triggers are included as arguments with ``Trigger`` argument role.

dueefin_w_tgg_woOtherType
    Similar to ``dueefin_wo_tgg_woOtherType``, triggers are included with **NO** additional entities.
    For analysis usage only.


.. _ChFinAnn: https://github.com/dolphin-zs/Doc2EDAG/blob/master/Data.zip
.. _DuEE-fin: https://aistudio.baidu.com/aistudio/competition/detail/46/0/task-definition
.. _Doc2EDAG: https://github.com/dolphin-zs/Doc2EDAG
.. _GIT: https://github.com/RunxinXu/GIT
