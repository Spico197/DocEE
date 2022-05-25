class BaseEvent(object):
    def __init__(self, fields, event_name="Event", key_fields=(), recguid=None):
        self.recguid = recguid
        self.name = event_name
        self.fields = list(fields)
        self.field2content = {f: None for f in fields}
        self.nonempty_count = 0
        self.nonempty_ratio = self.nonempty_count / len(self.fields)

        self.key_fields = set(key_fields)
        for key_field in self.key_fields:
            assert key_field in self.field2content

    def __repr__(self):
        event_str = "\n{}[\n".format(self.name)
        event_str += "  {}={}\n".format("recguid", self.recguid)
        event_str += "  {}={}\n".format("nonempty_count", self.nonempty_count)
        event_str += "  {}={:.3f}\n".format("nonempty_ratio", self.nonempty_ratio)
        event_str += "] (\n"
        for field in self.fields:
            if field in self.key_fields:
                key_str = " (key)"
            else:
                key_str = ""
            event_str += (
                "  "
                + field
                + "="
                + str(self.field2content[field])
                + ", {}\n".format(key_str)
            )
        event_str += ")\n"
        return event_str

    def update_by_dict(self, field2text, recguid=None):
        self.nonempty_count = 0
        self.recguid = recguid

        for field in self.fields:
            if field in field2text and field2text[field] is not None:
                self.nonempty_count += 1
                self.field2content[field] = field2text[field]
            else:
                self.field2content[field] = None

        self.nonempty_ratio = self.nonempty_count / len(self.fields)

    def field_to_dict(self):
        return dict(self.field2content)

    def set_key_fields(self, key_fields):
        self.key_fields = set(key_fields)

    def is_key_complete(self):
        for key_field in self.key_fields:
            if self.field2content[key_field] is None:
                return False

        return True

    def get_argument_tuple(self):
        args_tuple = tuple(self.field2content[field] for field in self.fields)
        return args_tuple

    def is_good_candidate(self, min_match_count=2):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityPledgeEvent(BaseEvent):
    NAME = "质押"
    FIELDS = [
        "Trigger",
        "质押物占总股比",
        "质权方",
        "质押方",
        "事件时间",
        "质押股票/股份数量",
        "质押物所属公司",
        "质押物",
        "质押物占持股比",
        "披露时间",
    ]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.3875
        2: ["Trigger", "质押股票/股份数量"],  # importance: 0.875
        3: ["Trigger", "事件时间", "质押股票/股份数量"],  # importance: 0.925
        4: ["Trigger", "事件时间", "质押物占持股比", "质押股票/股份数量"],  # importance: 0.95625
        5: ["Trigger", "质押物占总股比", "质押物占持股比", "质押股票/股份数量", "质权方"],  # importance: 0.975
        6: [
            "Trigger",
            "质押方",
            "质押物占总股比",
            "质押物占持股比",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.975
        7: [
            "Trigger",
            "事件时间",
            "质押方",
            "质押物占总股比",
            "质押物占持股比",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.975
        8: [
            "Trigger",
            "事件时间",
            "质押方",
            "质押物占总股比",
            "质押物占持股比",
            "质押物所属公司",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.975
        9: [
            "Trigger",
            "事件时间",
            "质押方",
            "质押物",
            "质押物占总股比",
            "质押物占持股比",
            "质押物所属公司",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.975
    }
    TRIGGERS["all"] = [
        "质押股票/股份数量",
        "质押物占持股比",
        "Trigger",
        "质押物占总股比",
        "事件时间",
        "质押方",
        "质押物",
        "披露时间",
        "质押物所属公司",
        "质权方",
    ]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ShareRepurchaseEvent(BaseEvent):
    NAME = "股份回购"
    FIELDS = [
        "Trigger",
        "每股交易价格",
        "交易金额",
        "回购完成时间",
        "回购股份数量",
        "占公司总股本比例",
        "回购方",
        "披露时间",
    ]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.720164609053498
        2: ["Trigger", "回购完成时间"],  # importance: 0.9670781893004116
        3: ["Trigger", "交易金额", "回购完成时间"],  # importance: 1.0
        4: ["Trigger", "交易金额", "回购完成时间", "每股交易价格"],  # importance: 1.0
        5: ["Trigger", "交易金额", "回购完成时间", "回购股份数量", "每股交易价格"],  # importance: 1.0
        6: [
            "Trigger",
            "交易金额",
            "占公司总股本比例",
            "回购完成时间",
            "回购股份数量",
            "每股交易价格",
        ],  # importance: 1.0
        7: [
            "Trigger",
            "交易金额",
            "占公司总股本比例",
            "回购完成时间",
            "回购方",
            "回购股份数量",
            "每股交易价格",
        ],  # importance: 1.0
        8: [
            "Trigger",
            "交易金额",
            "占公司总股本比例",
            "回购完成时间",
            "回购方",
            "回购股份数量",
            "披露时间",
            "每股交易价格",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = [
        "回购股份数量",
        "回购完成时间",
        "Trigger",
        "交易金额",
        "每股交易价格",
        "回购方",
        "占公司总股本比例",
        "披露时间",
    ]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ReleasePledgeEvent(BaseEvent):
    NAME = "解除质押"
    FIELDS = [
        "Trigger",
        "质权方",
        "质押物占总股比",
        "质押方",
        "事件时间",
        "质押股票/股份数量",
        "质押物所属公司",
        "质押物",
        "质押物占持股比",
        "披露时间",
    ]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.8220338983050848
        2: ["Trigger", "事件时间"],  # importance: 0.9491525423728814
        3: ["Trigger", "事件时间", "质权方"],  # importance: 0.9661016949152542
        4: ["Trigger", "事件时间", "披露时间", "质权方"],  # importance: 0.9745762711864406
        5: [
            "Trigger",
            "事件时间",
            "披露时间",
            "质押物占总股比",
            "质权方",
        ],  # importance: 0.9745762711864406
        6: [
            "Trigger",
            "事件时间",
            "披露时间",
            "质押方",
            "质押物占总股比",
            "质权方",
        ],  # importance: 0.9745762711864406
        7: [
            "Trigger",
            "事件时间",
            "披露时间",
            "质押方",
            "质押物占总股比",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.9745762711864406
        8: [
            "Trigger",
            "事件时间",
            "披露时间",
            "质押方",
            "质押物占总股比",
            "质押物所属公司",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.9745762711864406
        9: [
            "Trigger",
            "事件时间",
            "披露时间",
            "质押方",
            "质押物",
            "质押物占总股比",
            "质押物所属公司",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.9745762711864406
    }
    TRIGGERS["all"] = [
        "Trigger",
        "质押股票/股份数量",
        "事件时间",
        "质押方",
        "质押物",
        "披露时间",
        "质押物占持股比",
        "质押物所属公司",
        "质权方",
        "质押物占总股比",
    ]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class InvitedConversationEvent(BaseEvent):
    NAME = "被约谈"
    FIELDS = ["Trigger", "约谈机构", "被约谈时间", "披露时间", "公司名称"]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.34375
        2: ["Trigger", "公司名称"],  # importance: 1.0
        3: ["Trigger", "公司名称", "约谈机构"],  # importance: 1.0
        4: ["Trigger", "公司名称", "约谈机构", "被约谈时间"],  # importance: 1.0
        5: ["Trigger", "公司名称", "披露时间", "约谈机构", "被约谈时间"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["公司名称", "Trigger", "约谈机构", "被约谈时间", "披露时间"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class BusinessAcquisitionEvent(BaseEvent):
    NAME = "企业收购"
    FIELDS = ["Trigger", "被收购方", "收购标的", "交易金额", "收购方", "收购完成时间", "披露时间"]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.7887323943661971
        2: ["Trigger", "被收购方"],  # importance: 0.9507042253521126
        3: ["Trigger", "收购方", "被收购方"],  # importance: 0.9859154929577465
        4: ["Trigger", "交易金额", "收购方", "被收购方"],  # importance: 1.0
        5: ["Trigger", "交易金额", "收购方", "收购标的", "被收购方"],  # importance: 1.0
        6: ["Trigger", "交易金额", "收购完成时间", "收购方", "收购标的", "被收购方"],  # importance: 1.0
        7: [
            "Trigger",
            "交易金额",
            "披露时间",
            "收购完成时间",
            "收购方",
            "收购标的",
            "被收购方",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = ["被收购方", "收购方", "Trigger", "收购标的", "披露时间", "交易金额", "收购完成时间"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ShareholderOverweightEvent(BaseEvent):
    NAME = "股东增持"
    FIELDS = [
        "Trigger",
        "每股交易价格",
        "交易金额",
        "增持部分占所持比例",
        "交易完成时间",
        "增持方",
        "交易股票/股份数量",
        "增持部分占总股本比例",
        "股票简称",
        "披露时间",
    ]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.41935483870967744
        2: ["Trigger", "交易股票/股份数量"],  # importance: 0.8064516129032258
        3: ["Trigger", "交易股票/股份数量", "股票简称"],  # importance: 0.9032258064516129
        4: ["Trigger", "交易完成时间", "增持方", "股票简称"],  # importance: 0.9838709677419355
        5: ["Trigger", "交易完成时间", "交易金额", "增持方", "股票简称"],  # importance: 1.0
        6: ["Trigger", "交易完成时间", "交易金额", "增持方", "每股交易价格", "股票简称"],  # importance: 1.0
        7: [
            "Trigger",
            "交易完成时间",
            "交易金额",
            "增持方",
            "增持部分占所持比例",
            "每股交易价格",
            "股票简称",
        ],  # importance: 1.0
        8: [
            "Trigger",
            "交易完成时间",
            "交易股票/股份数量",
            "交易金额",
            "增持方",
            "增持部分占所持比例",
            "每股交易价格",
            "股票简称",
        ],  # importance: 1.0
        9: [
            "Trigger",
            "交易完成时间",
            "交易股票/股份数量",
            "交易金额",
            "增持方",
            "增持部分占总股本比例",
            "增持部分占所持比例",
            "每股交易价格",
            "股票简称",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = [
        "增持方",
        "股票简称",
        "交易股票/股份数量",
        "交易完成时间",
        "Trigger",
        "披露时间",
        "增持部分占总股本比例",
        "交易金额",
        "每股交易价格",
        "增持部分占所持比例",
    ]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ExecutivesChangeEvent(BaseEvent):
    NAME = "高管变动"
    FIELDS = [
        "Trigger",
        "变动后职位",
        "任职公司",
        "高管姓名",
        "披露日期",
        "变动类型",
        "事件时间",
        "高管职位",
        "变动后公司名称",
    ]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6044776119402985
        2: ["Trigger", "高管姓名"],  # importance: 0.9552238805970149
        3: ["Trigger", "变动类型", "高管姓名"],  # importance: 0.9701492537313433
        4: ["Trigger", "变动后职位", "变动类型", "高管姓名"],  # importance: 0.9701492537313433
        5: [
            "Trigger",
            "任职公司",
            "变动后职位",
            "变动类型",
            "高管姓名",
        ],  # importance: 0.9701492537313433
        6: [
            "Trigger",
            "任职公司",
            "变动后职位",
            "变动类型",
            "披露日期",
            "高管姓名",
        ],  # importance: 0.9701492537313433
        7: [
            "Trigger",
            "事件时间",
            "任职公司",
            "变动后职位",
            "变动类型",
            "披露日期",
            "高管姓名",
        ],  # importance: 0.9701492537313433
        8: [
            "Trigger",
            "事件时间",
            "任职公司",
            "变动后职位",
            "变动类型",
            "披露日期",
            "高管姓名",
            "高管职位",
        ],  # importance: 0.9701492537313433
        9: [
            "Trigger",
            "事件时间",
            "任职公司",
            "变动后公司名称",
            "变动后职位",
            "变动类型",
            "披露日期",
            "高管姓名",
            "高管职位",
        ],  # importance: 0.9701492537313433
    }
    TRIGGERS["all"] = [
        "高管姓名",
        "Trigger",
        "变动类型",
        "高管职位",
        "变动后职位",
        "任职公司",
        "事件时间",
        "披露日期",
        "变动后公司名称",
    ]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class WinBidEvent(BaseEvent):
    NAME = "中标"
    FIELDS = ["Trigger", "中标金额", "披露日期", "招标方", "中标日期", "中标标的", "中标公司"]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.835820895522388
        2: ["Trigger", "中标标的"],  # importance: 0.9552238805970149
        3: ["Trigger", "中标公司", "中标标的"],  # importance: 1.0
        4: ["Trigger", "中标公司", "中标标的", "中标金额"],  # importance: 1.0
        5: ["Trigger", "中标公司", "中标标的", "中标金额", "披露日期"],  # importance: 1.0
        6: ["Trigger", "中标公司", "中标日期", "中标金额", "披露日期", "招标方"],  # importance: 1.0
        7: [
            "Trigger",
            "中标公司",
            "中标日期",
            "中标标的",
            "中标金额",
            "披露日期",
            "招标方",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = ["中标标的", "中标公司", "Trigger", "中标金额", "中标日期", "招标方", "披露日期"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class CompanyIPOEvent(BaseEvent):
    NAME = "公司上市"
    FIELDS = ["Trigger", "募资金额", "事件时间", "证券代码", "环节", "发行价格", "上市公司", "披露时间", "市值"]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.7439024390243902
        2: ["Trigger", "上市公司"],  # importance: 0.9878048780487805
        3: ["Trigger", "上市公司", "募资金额"],  # importance: 0.9878048780487805
        4: ["Trigger", "上市公司", "事件时间", "募资金额"],  # importance: 0.9878048780487805
        5: [
            "Trigger",
            "上市公司",
            "事件时间",
            "募资金额",
            "证券代码",
        ],  # importance: 0.9878048780487805
        6: [
            "Trigger",
            "上市公司",
            "事件时间",
            "募资金额",
            "环节",
            "证券代码",
        ],  # importance: 0.9878048780487805
        7: [
            "Trigger",
            "上市公司",
            "事件时间",
            "募资金额",
            "发行价格",
            "环节",
            "证券代码",
        ],  # importance: 0.9878048780487805
        8: [
            "Trigger",
            "上市公司",
            "事件时间",
            "募资金额",
            "发行价格",
            "披露时间",
            "环节",
            "证券代码",
        ],  # importance: 0.9878048780487805
        9: [
            "Trigger",
            "上市公司",
            "事件时间",
            "募资金额",
            "发行价格",
            "市值",
            "披露时间",
            "环节",
            "证券代码",
        ],  # importance: 0.9878048780487805
    }
    TRIGGERS["all"] = [
        "上市公司",
        "Trigger",
        "事件时间",
        "披露时间",
        "证券代码",
        "募资金额",
        "发行价格",
        "市值",
        "环节",
    ]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class CompanyFinancingEvent(BaseEvent):
    NAME = "企业融资"
    FIELDS = ["Trigger", "融资金额", "事件时间", "被投资方", "领投方", "融资轮次", "披露时间", "投资方"]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6805555555555556
        2: ["Trigger", "融资金额"],  # importance: 0.9583333333333334
        3: ["Trigger", "事件时间", "融资金额"],  # importance: 0.9861111111111112
        4: ["Trigger", "事件时间", "披露时间", "融资金额"],  # importance: 1.0
        5: ["Trigger", "事件时间", "披露时间", "融资金额", "被投资方"],  # importance: 1.0
        6: ["Trigger", "事件时间", "披露时间", "融资金额", "被投资方", "领投方"],  # importance: 1.0
        7: [
            "Trigger",
            "事件时间",
            "披露时间",
            "融资轮次",
            "融资金额",
            "被投资方",
            "领投方",
        ],  # importance: 1.0
        8: [
            "Trigger",
            "事件时间",
            "投资方",
            "披露时间",
            "融资轮次",
            "融资金额",
            "被投资方",
            "领投方",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = ["融资金额", "被投资方", "Trigger", "投资方", "披露时间", "融资轮次", "领投方", "事件时间"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class CompanyLossEvent(BaseEvent):
    NAME = "亏损"
    FIELDS = ["Trigger", "亏损变化", "财报周期", "净亏损", "披露时间", "公司名称"]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6134969325153374
        2: ["Trigger", "净亏损"],  # importance: 1.0
        3: ["Trigger", "亏损变化", "净亏损"],  # importance: 1.0
        4: ["Trigger", "亏损变化", "净亏损", "财报周期"],  # importance: 1.0
        5: ["Trigger", "亏损变化", "净亏损", "披露时间", "财报周期"],  # importance: 1.0
        6: ["Trigger", "亏损变化", "公司名称", "净亏损", "披露时间", "财报周期"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["净亏损", "财报周期", "Trigger", "公司名称", "披露时间", "亏损变化"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ShareholderUnderweightEvent(BaseEvent):
    NAME = "股东减持"
    FIELDS = [
        "Trigger",
        "减持方",
        "每股交易价格",
        "交易金额",
        "减持部分占所持比例",
        "交易完成时间",
        "交易股票/股份数量",
        "减持部分占总股本比例",
        "股票简称",
        "披露时间",
    ]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.46258503401360546
        2: ["Trigger", "交易股票/股份数量"],  # importance: 0.782312925170068
        3: ["Trigger", "交易股票/股份数量", "股票简称"],  # importance: 0.9047619047619048
        4: [
            "Trigger",
            "交易股票/股份数量",
            "减持部分占总股本比例",
            "股票简称",
        ],  # importance: 0.9523809523809523
        5: [
            "Trigger",
            "交易完成时间",
            "交易股票/股份数量",
            "减持方",
            "股票简称",
        ],  # importance: 0.9863945578231292
        6: [
            "Trigger",
            "交易完成时间",
            "交易股票/股份数量",
            "减持方",
            "每股交易价格",
            "股票简称",
        ],  # importance: 0.9863945578231292
        7: [
            "Trigger",
            "交易完成时间",
            "交易股票/股份数量",
            "交易金额",
            "减持方",
            "每股交易价格",
            "股票简称",
        ],  # importance: 0.9863945578231292
        8: [
            "Trigger",
            "交易完成时间",
            "交易股票/股份数量",
            "交易金额",
            "减持方",
            "减持部分占所持比例",
            "每股交易价格",
            "股票简称",
        ],  # importance: 0.9863945578231292
        9: [
            "Trigger",
            "交易完成时间",
            "交易股票/股份数量",
            "交易金额",
            "减持方",
            "减持部分占总股本比例",
            "减持部分占所持比例",
            "每股交易价格",
            "股票简称",
        ],  # importance: 0.9863945578231292
    }
    TRIGGERS["all"] = [
        "减持方",
        "股票简称",
        "交易完成时间",
        "减持部分占总股本比例",
        "交易股票/股份数量",
        "Trigger",
        "披露时间",
        "交易金额",
        "每股交易价格",
        "减持部分占所持比例",
    ]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class CompanyBankruptEvent(BaseEvent):
    NAME = "企业破产"
    FIELDS = ["Trigger", "债务规模", "破产公司", "债权人", "破产时间", "披露时间"]

    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.75
        2: ["Trigger", "破产公司"],  # importance: 1.0
        3: ["Trigger", "债务规模", "破产公司"],  # importance: 1.0
        4: ["Trigger", "债务规模", "债权人", "破产公司"],  # importance: 1.0
        5: ["Trigger", "债务规模", "债权人", "破产公司", "破产时间"],  # importance: 1.0
        6: ["Trigger", "债务规模", "债权人", "披露时间", "破产公司", "破产时间"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["破产公司", "Trigger", "披露时间", "破产时间", "债务规模", "债权人"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


common_fields = ["OtherType"]


event_type2event_class = {
    EquityPledgeEvent.NAME: EquityPledgeEvent,
    ShareRepurchaseEvent.NAME: ShareRepurchaseEvent,
    ReleasePledgeEvent.NAME: ReleasePledgeEvent,
    InvitedConversationEvent.NAME: InvitedConversationEvent,
    BusinessAcquisitionEvent.NAME: BusinessAcquisitionEvent,
    ShareholderOverweightEvent.NAME: ShareholderOverweightEvent,
    ExecutivesChangeEvent.NAME: ExecutivesChangeEvent,
    WinBidEvent.NAME: WinBidEvent,
    CompanyIPOEvent.NAME: CompanyIPOEvent,
    CompanyFinancingEvent.NAME: CompanyFinancingEvent,
    CompanyLossEvent.NAME: CompanyLossEvent,
    ShareholderUnderweightEvent.NAME: ShareholderUnderweightEvent,
    CompanyBankruptEvent.NAME: CompanyBankruptEvent,
}


event_type_fields_list = [
    # name, fields, trigger fields, min_fields_num
    (EquityPledgeEvent.NAME, EquityPledgeEvent.FIELDS, EquityPledgeEvent.TRIGGERS, 2),
    (
        ShareRepurchaseEvent.NAME,
        ShareRepurchaseEvent.FIELDS,
        ShareRepurchaseEvent.TRIGGERS,
        2,
    ),
    (
        ReleasePledgeEvent.NAME,
        ReleasePledgeEvent.FIELDS,
        ReleasePledgeEvent.TRIGGERS,
        2,
    ),
    (
        InvitedConversationEvent.NAME,
        InvitedConversationEvent.FIELDS,
        InvitedConversationEvent.TRIGGERS,
        2,
    ),
    (
        BusinessAcquisitionEvent.NAME,
        BusinessAcquisitionEvent.FIELDS,
        BusinessAcquisitionEvent.TRIGGERS,
        2,
    ),
    (
        ShareholderOverweightEvent.NAME,
        ShareholderOverweightEvent.FIELDS,
        ShareholderOverweightEvent.TRIGGERS,
        2,
    ),
    (
        ExecutivesChangeEvent.NAME,
        ExecutivesChangeEvent.FIELDS,
        ExecutivesChangeEvent.TRIGGERS,
        2,
    ),
    (WinBidEvent.NAME, WinBidEvent.FIELDS, WinBidEvent.TRIGGERS, 2),
    (CompanyIPOEvent.NAME, CompanyIPOEvent.FIELDS, CompanyIPOEvent.TRIGGERS, 2),
    # (CompanyIPOEvent.NAME, CompanyIPOEvent.FIELDS, CompanyIPOEvent.TRIGGERS, 2),
    (
        CompanyFinancingEvent.NAME,
        CompanyFinancingEvent.FIELDS,
        CompanyFinancingEvent.TRIGGERS,
        2,
    ),
    (CompanyLossEvent.NAME, CompanyLossEvent.FIELDS, CompanyLossEvent.TRIGGERS, 2),
    (
        ShareholderUnderweightEvent.NAME,
        ShareholderUnderweightEvent.FIELDS,
        ShareholderUnderweightEvent.TRIGGERS,
        2,
    ),
    (
        CompanyBankruptEvent.NAME,
        CompanyBankruptEvent.FIELDS,
        CompanyBankruptEvent.TRIGGERS,
        2,
    ),
]
