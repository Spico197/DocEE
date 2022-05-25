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
        1: ["质押股票/股份数量"],  # importance: 0.7726953125
        2: ["事件时间", "质押股票/股份数量"],  # importance: 0.859765625
        3: ["事件时间", "质押方", "质押股票/股份数量"],  # importance: 0.9125
        4: ["事件时间", "质押方", "质押物占总股比", "质押股票/股份数量"],  # importance: 0.93125
        5: ["事件时间", "披露时间", "质押物占总股比", "质押物占持股比", "质押股票/股份数量"],  # importance: 0.95
        6: [
            "披露时间",
            "质押方",
            "质押物占总股比",
            "质押物占持股比",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.95
        7: [
            "事件时间",
            "披露时间",
            "质押方",
            "质押物占总股比",
            "质押物占持股比",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.95
        8: [
            "事件时间",
            "披露时间",
            "质押方",
            "质押物占总股比",
            "质押物占持股比",
            "质押物所属公司",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.95
        9: [
            "事件时间",
            "披露时间",
            "质押方",
            "质押物",
            "质押物占总股比",
            "质押物占持股比",
            "质押物所属公司",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.95
    }
    TRIGGERS["all"] = [
        "质押股票/股份数量",
        "质押物占持股比",
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
    FIELDS = ["每股交易价格", "交易金额", "回购完成时间", "回购股份数量", "占公司总股本比例", "回购方", "披露时间"]

    TRIGGERS = {
        1: ["回购股份数量"],  # importance: 0.8759843519788649
        2: ["交易金额", "回购股份数量"],  # importance: 0.9713627665159444
        3: ["交易金额", "回购完成时间", "回购方"],  # importance: 0.9958847736625515
        4: ["回购完成时间", "回购方", "回购股份数量", "每股交易价格"],  # importance: 1.0
        5: ["交易金额", "回购完成时间", "回购方", "回购股份数量", "每股交易价格"],  # importance: 1.0
        6: ["交易金额", "占公司总股本比例", "回购完成时间", "回购方", "回购股份数量", "每股交易价格"],  # importance: 1.0
        7: [
            "交易金额",
            "占公司总股本比例",
            "回购完成时间",
            "回购方",
            "回购股份数量",
            "披露时间",
            "每股交易价格",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = ["回购股份数量", "回购完成时间", "交易金额", "每股交易价格", "回购方", "占公司总股本比例", "披露时间"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ReleasePledgeEvent(BaseEvent):
    NAME = "解除质押"
    FIELDS = [
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
        1: ["质押股票/股份数量"],  # importance: 0.7436799770180982
        2: ["事件时间", "质押股票/股份数量"],  # importance: 0.8514794599253088
        3: ["事件时间", "质押物", "质押股票/股份数量"],  # importance: 0.8983050847457628
        4: ["事件时间", "披露时间", "质押物", "质押股票/股份数量"],  # importance: 0.9067796610169492
        5: [
            "事件时间",
            "披露时间",
            "质押物",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.9067796610169492
        6: [
            "事件时间",
            "披露时间",
            "质押物",
            "质押物占总股比",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.9067796610169492
        7: [
            "事件时间",
            "披露时间",
            "质押方",
            "质押物",
            "质押物占总股比",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.9067796610169492
        8: [
            "事件时间",
            "披露时间",
            "质押方",
            "质押物",
            "质押物占总股比",
            "质押物所属公司",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.9067796610169492
        9: [
            "事件时间",
            "披露时间",
            "质押方",
            "质押物",
            "质押物占总股比",
            "质押物占持股比",
            "质押物所属公司",
            "质押股票/股份数量",
            "质权方",
        ],  # importance: 0.9067796610169492
    }
    TRIGGERS["all"] = [
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
    FIELDS = ["约谈机构", "被约谈时间", "披露时间", "公司名称"]

    TRIGGERS = {
        1: ["公司名称"],  # importance: 0.9375
        2: ["公司名称", "约谈机构"],  # importance: 1.0
        3: ["公司名称", "约谈机构", "被约谈时间"],  # importance: 1.0
        4: ["公司名称", "披露时间", "约谈机构", "被约谈时间"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["公司名称", "约谈机构", "被约谈时间", "披露时间"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class BusinessAcquisitionEvent(BaseEvent):
    NAME = "企业收购"
    TRIGGERS = ["被收购方", "收购方", "收购标的", "披露时间", "收购完成时间", "交易金额"]
    FIELDS = ["被收购方", "收购标的", "交易金额", "收购方", "收购完成时间", "披露时间"]

    TRIGGERS = {
        1: ["被收购方"],  # importance: 0.8961515572307083
        2: ["收购方", "被收购方"],  # importance: 0.9647887323943662
        3: ["交易金额", "收购方", "被收购方"],  # importance: 0.9929577464788732
        4: ["交易金额", "收购方", "收购标的", "被收购方"],  # importance: 1.0
        5: ["交易金额", "收购完成时间", "收购方", "收购标的", "被收购方"],  # importance: 1.0
        6: ["交易金额", "披露时间", "收购完成时间", "收购方", "收购标的", "被收购方"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["被收购方", "收购方", "收购标的", "披露时间", "交易金额", "收购完成时间"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ShareholderOverweightEvent(BaseEvent):
    NAME = "股东增持"
    TRIGGERS = [
        "交易股票/股份数量",
        "股票简称",
        "交易金额",
        "增持方",
        "交易完成时间",
        "增持部分占总股本比例",
        "每股交易价格",
        "披露时间",
        "增持部分占所持比例",
    ]
    FIELDS = [
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
        1: ["增持方"],  # importance: 0.6399583766909469
        2: ["交易股票/股份数量", "股票简称"],  # importance: 0.9032258064516129
        3: ["交易完成时间", "增持方", "股票简称"],  # importance: 0.9838709677419355
        4: ["交易完成时间", "交易金额", "增持方", "股票简称"],  # importance: 1.0
        5: ["交易完成时间", "交易金额", "增持方", "每股交易价格", "股票简称"],  # importance: 1.0
        6: ["交易完成时间", "交易金额", "增持方", "增持部分占所持比例", "每股交易价格", "股票简称"],  # importance: 1.0
        7: [
            "交易完成时间",
            "交易股票/股份数量",
            "交易金额",
            "增持方",
            "增持部分占所持比例",
            "每股交易价格",
            "股票简称",
        ],  # importance: 1.0
        8: [
            "交易完成时间",
            "交易股票/股份数量",
            "交易金额",
            "增持方",
            "增持部分占总股本比例",
            "增持部分占所持比例",
            "每股交易价格",
            "股票简称",
        ],  # importance: 1.0
        9: [
            "交易完成时间",
            "交易股票/股份数量",
            "交易金额",
            "增持方",
            "增持部分占总股本比例",
            "增持部分占所持比例",
            "披露时间",
            "每股交易价格",
            "股票简称",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = [
        "增持方",
        "股票简称",
        "交易股票/股份数量",
        "交易完成时间",
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
    TRIGGERS = ["高管姓名", "变动类型", "变动后职位", "高管职位", "事件时间", "披露日期", "任职公司", "变动后公司名称"]
    FIELDS = ["变动后职位", "任职公司", "高管姓名", "披露日期", "变动类型", "事件时间", "高管职位", "变动后公司名称"]

    TRIGGERS = {
        1: ["高管姓名"],  # importance: 0.9328358208955224
        2: ["变动类型", "高管姓名"],  # importance: 0.9701492537313433
        3: ["变动后职位", "变动类型", "高管姓名"],  # importance: 0.9776119402985075
        4: ["任职公司", "变动后职位", "变动类型", "高管姓名"],  # importance: 0.9776119402985075
        5: ["任职公司", "变动后职位", "变动类型", "披露日期", "高管姓名"],  # importance: 0.9776119402985075
        6: [
            "事件时间",
            "任职公司",
            "变动后职位",
            "变动类型",
            "披露日期",
            "高管姓名",
        ],  # importance: 0.9776119402985075
        7: [
            "事件时间",
            "任职公司",
            "变动后职位",
            "变动类型",
            "披露日期",
            "高管姓名",
            "高管职位",
        ],  # importance: 0.9776119402985075
        8: [
            "事件时间",
            "任职公司",
            "变动后公司名称",
            "变动后职位",
            "变动类型",
            "披露日期",
            "高管姓名",
            "高管职位",
        ],  # importance: 0.9776119402985075
    }
    TRIGGERS["all"] = [
        "高管姓名",
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
    TRIGGERS = ["中标标的", "中标公司", "中标金额", "招标方", "披露日期", "中标日期"]
    FIELDS = ["中标金额", "披露日期", "招标方", "中标日期", "中标标的", "中标公司"]

    TRIGGERS = {
        1: ["中标标的"],  # importance: 0.9194698151035865
        2: ["中标公司", "中标标的"],  # importance: 1.0
        3: ["中标公司", "中标标的", "中标金额"],  # importance: 1.0
        4: ["中标公司", "中标标的", "中标金额", "披露日期"],  # importance: 1.0
        5: ["中标公司", "中标日期", "中标金额", "披露日期", "招标方"],  # importance: 1.0
        6: ["中标公司", "中标日期", "中标标的", "中标金额", "披露日期", "招标方"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["中标标的", "中标公司", "中标金额", "中标日期", "招标方", "披露日期"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class CompanyIPOEvent(BaseEvent):
    NAME = "公司上市"
    TRIGGERS = ["环节", "上市公司", "事件时间", "证券代码", "披露时间", "市值", "发行价格", "募资金额"]
    FIELDS = ["募资金额", "事件时间", "证券代码", "环节", "发行价格", "上市公司", "披露时间", "市值"]

    TRIGGERS = {
        1: ["上市公司"],  # importance: 0.8902439024390244
        2: ["上市公司", "事件时间"],  # importance: 0.975609756097561
        3: ["上市公司", "事件时间", "募资金额"],  # importance: 0.975609756097561
        4: ["上市公司", "事件时间", "募资金额", "证券代码"],  # importance: 0.975609756097561
        5: ["上市公司", "事件时间", "募资金额", "环节", "证券代码"],  # importance: 0.975609756097561
        6: [
            "上市公司",
            "事件时间",
            "募资金额",
            "发行价格",
            "环节",
            "证券代码",
        ],  # importance: 0.975609756097561
        7: [
            "上市公司",
            "事件时间",
            "募资金额",
            "发行价格",
            "披露时间",
            "环节",
            "证券代码",
        ],  # importance: 0.975609756097561
        8: [
            "上市公司",
            "事件时间",
            "募资金额",
            "发行价格",
            "市值",
            "披露时间",
            "环节",
            "证券代码",
        ],  # importance: 0.975609756097561
    }
    TRIGGERS["all"] = ["上市公司", "事件时间", "披露时间", "证券代码", "募资金额", "发行价格", "市值", "环节"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class CompanyFinancingEvent(BaseEvent):
    NAME = "企业融资"
    TRIGGERS = ["融资金额", "融资轮次", "被投资方", "事件时间", "投资方", "领投方", "披露时间"]
    FIELDS = ["融资金额", "事件时间", "被投资方", "领投方", "融资轮次", "披露时间", "投资方"]

    TRIGGERS = {
        1: ["融资金额"],  # importance: 0.8013117283950618
        2: ["投资方", "融资金额"],  # importance: 0.931712962962963
        3: ["事件时间", "披露时间", "融资金额"],  # importance: 0.9724151234567903
        4: ["事件时间", "披露时间", "融资金额", "领投方"],  # importance: 1.0
        5: ["事件时间", "披露时间", "融资金额", "被投资方", "领投方"],  # importance: 1.0
        6: ["事件时间", "披露时间", "融资轮次", "融资金额", "被投资方", "领投方"],  # importance: 1.0
        7: ["事件时间", "投资方", "披露时间", "融资轮次", "融资金额", "被投资方", "领投方"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["融资金额", "被投资方", "投资方", "披露时间", "融资轮次", "领投方", "事件时间"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class CompanyLossEvent(BaseEvent):
    NAME = "亏损"
    TRIGGERS = ["净亏损", "财报周期", "公司名称", "披露时间", "亏损变化"]
    FIELDS = ["亏损变化", "财报周期", "净亏损", "披露时间", "公司名称"]

    TRIGGERS = {
        1: ["净亏损"],  # importance: 0.9815950920245399
        2: ["净亏损", "财报周期"],  # importance: 1.0
        3: ["亏损变化", "净亏损", "财报周期"],  # importance: 1.0
        4: ["亏损变化", "净亏损", "披露时间", "财报周期"],  # importance: 1.0
        5: ["亏损变化", "公司名称", "净亏损", "披露时间", "财报周期"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["净亏损", "财报周期", "公司名称", "披露时间", "亏损变化"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ShareholderUnderweightEvent(BaseEvent):
    NAME = "股东减持"
    TRIGGERS = [
        "交易股票/股份数量",
        "股票简称",
        "减持方",
        "减持部分占总股本比例",
        "披露时间",
        "交易金额",
        "交易完成时间",
        "每股交易价格",
        "减持部分占所持比例",
    ]
    FIELDS = [
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
        1: ["减持方"],  # importance: 0.5945670785320931
        2: ["交易股票/股份数量", "股票简称"],  # importance: 0.8729695960016659
        3: ["交易股票/股份数量", "减持方", "股票简称"],  # importance: 0.9455782312925171
        4: ["交易完成时间", "交易股票/股份数量", "减持方", "股票简称"],  # importance: 0.9863945578231292
        5: [
            "交易完成时间",
            "交易股票/股份数量",
            "减持方",
            "每股交易价格",
            "股票简称",
        ],  # importance: 0.9863945578231292
        6: [
            "交易完成时间",
            "交易股票/股份数量",
            "交易金额",
            "减持方",
            "每股交易价格",
            "股票简称",
        ],  # importance: 0.9863945578231292
        7: [
            "交易完成时间",
            "交易股票/股份数量",
            "交易金额",
            "减持方",
            "减持部分占所持比例",
            "每股交易价格",
            "股票简称",
        ],  # importance: 0.9863945578231292
        8: [
            "交易完成时间",
            "交易股票/股份数量",
            "交易金额",
            "减持方",
            "减持部分占总股本比例",
            "减持部分占所持比例",
            "每股交易价格",
            "股票简称",
        ],  # importance: 0.9863945578231292
        9: [
            "交易完成时间",
            "交易股票/股份数量",
            "交易金额",
            "减持方",
            "减持部分占总股本比例",
            "减持部分占所持比例",
            "披露时间",
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
    TRIGGERS = ["破产公司", "披露时间", "披露时间", "债务规模", "债权人"]
    FIELDS = ["债务规模", "破产公司", "债权人", "破产时间", "披露时间"]

    TRIGGERS = {
        1: ["破产公司"],  # importance: 0.9090909090909091
        2: ["披露时间", "破产公司"],  # importance: 0.9545454545454546
        3: ["债务规模", "披露时间", "破产公司"],  # importance: 0.9772727272727273
        4: ["债务规模", "债权人", "披露时间", "破产公司"],  # importance: 0.9772727272727273
        5: ["债务规模", "债权人", "披露时间", "破产公司", "破产时间"],  # importance: 0.9772727272727273
    }
    TRIGGERS["all"] = ["破产公司", "披露时间", "破产时间", "债务规模", "债权人"]

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
