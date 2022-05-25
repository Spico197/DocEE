import os

from dee.tasks import DEETask, DEETaskSetting

if __name__ == "__main__":
    # init
    task_dir = "Exps/sct-Tp1CG-with_left_trigger-OtherType-comp_ents-bs64_8"
    cpt_file_name = "TriggerAwarePrunedCompleteGraph"
    # bert_model_dir is for tokenization use, `vocab.txt` must be included in this dir
    # change this to `bert-base-chinese` to use the huggingface online cache
    bert_model_dir = "/path/to/bert-base-chinese"

    # load settings
    dee_setting = DEETaskSetting.from_pretrained(
        os.path.join(task_dir, f"{cpt_file_name}.task_setting.json")
    )
    dee_setting.local_rank = -1
    dee_setting.filtered_data_types = "o2o,o2m,m2m,unk"
    dee_setting.bert_model = bert_model_dir

    # build task
    dee_task = DEETask(
        dee_setting,
        load_train=False,
        load_dev=False,
        load_test=False,
        load_inference=False,
        parallel_decorate=False,
    )

    # load PTPCG parameters
    dee_task.resume_cpt_at(57)

    # predict
    doc = (
        "证券代码：300142证券简称：沃森生物公告编号：2016-072"
        "云南沃森生物技术股份有限公司关于股东解除股权质押的公告"
        "本公司及董事会全体成员保证信息披露内容的真实、准确和完整，没有虚假记载、误导性陈述或重大遗漏。"
        "云南沃森生物技术股份有限公司（以下简称“公司”）日前接到股东李云春先生函告，获悉李云春先生所持有的本公司部分股份解除质押，具体情况如下："
        "李云春先生曾于2015年5月7日同招商证券股份有限公司就其持有的13596398股公司股票办理了股票质押回购业务，质押期限自2015年5月7日起"
        "至质权人向中国证券登记结算有限责任公司深圳分公司办理解除质押登记为止（详见公司在证监会指定的信息披露网站巨潮资讯网披露的第2015-048号公告）。"
        "李云春先生于2016年5月6日在中国证券登记结算有限责任公司深圳分公司办理了40789194股（含除权后派送的27192796股股份）公司股份的质押解除手续。"
        "李云春先生本次解除质押的公司股份占公司股份总数的2.91%，占其所持公司股份的25.16%。"
        "截至本公告披露日，李云春先生共持有公司股份162103218股，占公司股份总数的11.55%。"
        "李云春先生共质押其持有的公司股份86304393股，占公司股份总数的6.15%，占其所持公司股份的53.24%。"
        "特此公告。"
        "云南沃森生物技术股份有限公司"
        "董事会"
        "二〇一六年五月九日"
    )
    results = dee_task.predict_one(doc)
    print(results)
