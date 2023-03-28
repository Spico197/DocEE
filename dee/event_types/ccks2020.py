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


class BankruptcyEvent(BaseEvent):
    NAME = "破产清算"
    FIELDS = ["公司名称", "公告时间", "受理法院", "裁定时间", "公司行业"]
    TRIGGERS = {
        1: ["公司名称"],  # importance: 0.9950641658440277
        2: ["公司名称", "公告时间"],  # importance: 0.9990128331688055
        3: ["公司名称", "公告时间", "受理法院"],  # importance: 1.0
        4: ["公司名称", "公司行业", "公告时间", "受理法院"],  # importance: 1.0
        5: ["公司名称", "公司行业", "公告时间", "受理法院", "裁定时间"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["公司名称", "公告时间", "受理法院", "裁定时间", "公司行业"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class AccidentEvent(BaseEvent):
    NAME = "重大安全事故"
    FIELDS = ["公司名称", "公告时间", "伤亡人数", "损失金额", "其他影响"]
    TRIGGERS = {
        1: ["公司名称"],  # importance: 0.9974424552429667
        2: ["公司名称", "公告时间"],  # importance: 1.0
        3: ["伤亡人数", "公司名称", "公告时间"],  # importance: 1.0
        4: ["伤亡人数", "公司名称", "公告时间", "损失金额"],  # importance: 1.0
        5: ["伤亡人数", "公司名称", "公告时间", "其他影响", "损失金额"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["公司名称", "公告时间", "伤亡人数", "损失金额", "其他影响"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class EquityUnderweightEvent(BaseEvent):
    NAME = "股东减持"
    FIELDS = ["减持金额", "减持开始日期", "减持的股东"]
    TRIGGERS = {
        1: ["减持金额"],  # importance: 0.9486062717770035
        2: ["减持开始日期", "减持金额"],  # importance: 0.9817073170731707
        3: ["减持开始日期", "减持的股东", "减持金额"],  # importance: 0.990418118466899
    }
    TRIGGERS["all"] = ["减持金额", "减持开始日期", "减持的股东"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class EquityPledgeEvent(BaseEvent):
    NAME = "股权质押"
    FIELDS = ["质押金额", "质押开始日期", "接收方", "质押方", "质押结束日期"]
    TRIGGERS = {
        1: ["质押金额"],  # importance: 0.9625668449197861
        2: ["质押开始日期", "质押金额"],  # importance: 0.9910873440285205
        3: ["接收方", "质押开始日期", "质押金额"],  # importance: 0.9964349376114082
        4: ["接收方", "质押开始日期", "质押结束日期", "质押金额"],  # importance: 1.0
        5: ["接收方", "质押开始日期", "质押方", "质押结束日期", "质押金额"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["质押金额", "质押开始日期", "接收方", "质押方", "质押结束日期"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class EquityOverweightEvent(BaseEvent):
    NAME = "股东增持"
    FIELDS = ["增持金额", "增持开始日期", "增持的股东"]
    TRIGGERS = {
        1: ["增持金额"],  # importance: 0.9607609988109393
        2: ["增持的股东", "增持金额"],  # importance: 0.9892984542211652
        3: ["增持开始日期", "增持的股东", "增持金额"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["增持金额", "增持开始日期", "增持的股东"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class EquityFreezeEvent(BaseEvent):
    NAME = "股权冻结"
    FIELDS = ["冻结金额", "冻结开始日期", "被冻结股东", "冻结结束日期"]
    TRIGGERS = {
        1: ["冻结金额"],  # importance: 0.8524822695035461
        2: ["冻结开始日期", "冻结金额"],  # importance: 0.9687943262411347
        3: ["冻结开始日期", "冻结金额", "被冻结股东"],  # importance: 0.9716312056737588
        4: ["冻结开始日期", "冻结结束日期", "冻结金额", "被冻结股东"],  # importance: 0.9730496453900709
    }
    TRIGGERS["all"] = ["冻结金额", "冻结开始日期", "被冻结股东", "冻结结束日期"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class LeaderDeathEvent(BaseEvent):
    NAME = "高层死亡"
    FIELDS = ["公司名称", "高层人员", "高层职务", "死亡/失联时间", "死亡年龄"]
    TRIGGERS = {
        1: ["公司名称"],  # importance: 1.0
        2: ["公司名称", "高层人员"],  # importance: 1.0
        3: ["公司名称", "高层人员", "高层职务"],  # importance: 1.0
        4: ["公司名称", "死亡/失联时间", "高层人员", "高层职务"],  # importance: 1.0
        5: ["公司名称", "死亡/失联时间", "死亡年龄", "高层人员", "高层职务"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["公司名称", "高层人员", "高层职务", "死亡/失联时间", "死亡年龄"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class AssetLossEvent(BaseEvent):
    NAME = "重大资产损失"
    FIELDS = ["公司名称", "公告时间", "损失金额", "其他损失"]
    TRIGGERS = {
        1: ["公司名称"],  # importance: 0.9949494949494949
        2: ["公司名称", "公告时间"],  # importance: 1.0
        3: ["公司名称", "公告时间", "损失金额"],  # importance: 1.0
        4: ["公司名称", "公告时间", "其他损失", "损失金额"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["公司名称", "公告时间", "损失金额", "其他损失"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ExternalIndemnityEvent(BaseEvent):
    NAME = "重大对外赔付"
    FIELDS = ["公告时间", "公司名称", "赔付对象", "赔付金额"]
    TRIGGERS = {
        1: ["公告时间"],  # importance: 0.984251968503937
        2: ["公司名称", "公告时间"],  # importance: 1.0
        3: ["公司名称", "公告时间", "赔付对象"],  # importance: 1.0
        4: ["公司名称", "公告时间", "赔付对象", "赔付金额"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["公告时间", "公司名称", "赔付对象", "赔付金额"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


common_fields = []
event_type2event_class = {
    BankruptcyEvent.NAME: BankruptcyEvent,
    AccidentEvent.NAME: AccidentEvent,
    EquityUnderweightEvent.NAME: EquityUnderweightEvent,
    EquityPledgeEvent.NAME: EquityPledgeEvent,
    EquityOverweightEvent.NAME: EquityOverweightEvent,
    EquityFreezeEvent.NAME: EquityFreezeEvent,
    LeaderDeathEvent.NAME: LeaderDeathEvent,
    AssetLossEvent.NAME: AssetLossEvent,
    ExternalIndemnityEvent.NAME: ExternalIndemnityEvent,
}
event_type_fields_list = [
    (BankruptcyEvent.NAME, BankruptcyEvent.FIELDS, BankruptcyEvent.TRIGGERS, 2),
    (AccidentEvent.NAME, AccidentEvent.FIELDS, AccidentEvent.TRIGGERS, 2),
    (
        EquityUnderweightEvent.NAME,
        EquityUnderweightEvent.FIELDS,
        EquityUnderweightEvent.TRIGGERS,
        2,
    ),
    (EquityPledgeEvent.NAME, EquityPledgeEvent.FIELDS, EquityPledgeEvent.TRIGGERS, 2),
    (
        EquityOverweightEvent.NAME,
        EquityOverweightEvent.FIELDS,
        EquityOverweightEvent.TRIGGERS,
        2,
    ),
    (EquityFreezeEvent.NAME, EquityFreezeEvent.FIELDS, EquityFreezeEvent.TRIGGERS, 2),
    (LeaderDeathEvent.NAME, LeaderDeathEvent.FIELDS, LeaderDeathEvent.TRIGGERS, 2),
    (AssetLossEvent.NAME, AssetLossEvent.FIELDS, AssetLossEvent.TRIGGERS, 2),
    (
        ExternalIndemnityEvent.NAME,
        ExternalIndemnityEvent.FIELDS,
        ExternalIndemnityEvent.TRIGGERS,
        2,
    ),
]
