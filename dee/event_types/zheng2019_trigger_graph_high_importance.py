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

    def is_good_candidate(self):
        raise NotImplementedError()

    def get_argument_tuple(self):
        args_tuple = tuple(self.field2content[field] for field in self.fields)
        return args_tuple


class EquityFreezeEvent(BaseEvent):
    NAME = "EquityFreeze"
    # TRIGGERS = ['LegalInstitution', 'FrozeShares', 'StartDate', 'EquityHolder', 'TotalHoldingRatio', 'UnfrozeDate', 'EndDate', 'TotalHoldingShares']
    TRIGGERS = {
        1: ["LegalInstitution"],  # importance: 0.706060606060606
        2: ["FrozeShares", "LegalInstitution"],  # importance: 0.8757575757575757
        3: [
            "FrozeShares",
            "LegalInstitution",
            "StartDate",
        ],  # importance: 0.9242424242424242
        4: [
            "EquityHolder",
            "FrozeShares",
            "LegalInstitution",
            "StartDate",
        ],  # importance: 0.9303030303030303
        5: [
            "EquityHolder",
            "FrozeShares",
            "LegalInstitution",
            "StartDate",
            "TotalHoldingRatio",
        ],  # importance: 0.9333333333333333
        6: [
            "EquityHolder",
            "FrozeShares",
            "LegalInstitution",
            "StartDate",
            "TotalHoldingRatio",
            "UnfrozeDate",
        ],  # importance: 0.9363636363636364
        7: [
            "EquityHolder",
            "FrozeShares",
            "LegalInstitution",
            "StartDate",
            "TotalHoldingRatio",
            "TotalHoldingShares",
            "UnfrozeDate",
        ],  # importance: 0.9363636363636364
        8: [
            "EndDate",
            "EquityHolder",
            "FrozeShares",
            "LegalInstitution",
            "StartDate",
            "TotalHoldingRatio",
            "TotalHoldingShares",
            "UnfrozeDate",
        ],  # importance: 0.9363636363636364
    }
    TRIGGERS["all"] = [
        "LegalInstitution",
        "FrozeShares",
        "EquityHolder",
        "TotalHoldingShares",
        "StartDate",
        "TotalHoldingRatio",
        "EndDate",
        "UnfrozeDate",
    ]
    FIELDS = [
        "EquityHolder",
        "FrozeShares",
        "LegalInstitution",
        "TotalHoldingShares",
        "TotalHoldingRatio",
        "StartDate",
        "EndDate",
        "UnfrozeDate",
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EquityFreezeEvent.FIELDS, event_name=EquityFreezeEvent.NAME, recguid=recguid
        )
        self.set_key_fields(
            [
                "EquityHolder",
                "FrozeShares",
                "LegalInstitution",
            ]
        )

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityRepurchaseEvent(BaseEvent):
    NAME = "EquityRepurchase"
    # TRIGGERS = ['RepurchasedShares', 'ClosingDate', 'RepurchaseAmount', 'LowestTradingPrice', 'CompanyName', 'HighestTradingPrice']\
    TRIGGERS = {
        1: ["RepurchasedShares"],  # importance: 0.9718339642070036
        2: ["RepurchaseAmount", "RepurchasedShares"],  # importance: 0.9918962722852512
        3: ["ClosingDate", "RepurchaseAmount", "RepurchasedShares"],  # importance: 1.0
        4: [
            "ClosingDate",
            "CompanyName",
            "RepurchaseAmount",
            "RepurchasedShares",
        ],  # importance: 1.0
        5: [
            "ClosingDate",
            "CompanyName",
            "HighestTradingPrice",
            "RepurchaseAmount",
            "RepurchasedShares",
        ],  # importance: 1.0
        6: [
            "ClosingDate",
            "CompanyName",
            "HighestTradingPrice",
            "LowestTradingPrice",
            "RepurchaseAmount",
            "RepurchasedShares",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = [
        "RepurchasedShares",
        "LowestTradingPrice",
        "HighestTradingPrice",
        "CompanyName",
        "RepurchaseAmount",
        "ClosingDate",
    ]
    FIELDS = [
        "CompanyName",
        "HighestTradingPrice",
        "LowestTradingPrice",
        "RepurchasedShares",
        "ClosingDate",
        "RepurchaseAmount",
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EquityRepurchaseEvent.FIELDS,
            event_name=EquityRepurchaseEvent.NAME,
            recguid=recguid,
        )
        self.set_key_fields(
            [
                "CompanyName",
            ]
        )

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityUnderweightEvent(BaseEvent):
    NAME = "EquityUnderweight"
    # TRIGGERS = ['TradedShares', 'EquityHolder', 'StartDate', 'LaterHoldingShares', 'AveragePrice', 'EndDate']
    TRIGGERS = {
        1: ["TradedShares"],  # importance: 0.918848167539267
        2: ["EndDate", "EquityHolder"],  # importance: 0.9973821989528796
        3: [
            "EquityHolder",
            "StartDate",
            "TradedShares",
        ],  # importance: 0.9973821989528796
        4: [
            "EndDate",
            "EquityHolder",
            "StartDate",
            "TradedShares",
        ],  # importance: 0.9973821989528796
        5: [
            "EndDate",
            "EquityHolder",
            "LaterHoldingShares",
            "StartDate",
            "TradedShares",
        ],  # importance: 0.9973821989528796
        6: [
            "AveragePrice",
            "EndDate",
            "EquityHolder",
            "LaterHoldingShares",
            "StartDate",
            "TradedShares",
        ],  # importance: 0.9973821989528796
    }
    TRIGGERS["all"] = [
        "TradedShares",
        "EndDate",
        "StartDate",
        "EquityHolder",
        "LaterHoldingShares",
        "AveragePrice",
    ]
    FIELDS = [
        "EquityHolder",
        "TradedShares",
        "StartDate",
        "EndDate",
        "LaterHoldingShares",
        "AveragePrice",
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EquityUnderweightEvent.FIELDS,
            event_name=EquityUnderweightEvent.NAME,
            recguid=recguid,
        )
        self.set_key_fields(
            [
                "EquityHolder",
                "TradedShares",
            ]
        )

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityOverweightEvent(BaseEvent):
    NAME = "EquityOverweight"
    # TRIGGERS = ['TradedShares', 'EquityHolder', 'EndDate', 'LaterHoldingShares', 'AveragePrice', 'StartDate']
    TRIGGERS = {
        1: ["TradedShares"],  # importance: 0.9325396825396826
        2: ["EquityHolder", "StartDate"],  # importance: 0.996031746031746
        3: [
            "EquityHolder",
            "StartDate",
            "TradedShares",
        ],  # importance: 0.998015873015873
        4: [
            "EquityHolder",
            "LaterHoldingShares",
            "StartDate",
            "TradedShares",
        ],  # importance: 1.0
        5: [
            "EndDate",
            "EquityHolder",
            "LaterHoldingShares",
            "StartDate",
            "TradedShares",
        ],  # importance: 1.0
        6: [
            "AveragePrice",
            "EndDate",
            "EquityHolder",
            "LaterHoldingShares",
            "StartDate",
            "TradedShares",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = [
        "TradedShares",
        "StartDate",
        "EndDate",
        "EquityHolder",
        "LaterHoldingShares",
        "AveragePrice",
    ]
    FIELDS = [
        "EquityHolder",
        "TradedShares",
        "StartDate",
        "EndDate",
        "LaterHoldingShares",
        "AveragePrice",
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EquityOverweightEvent.FIELDS,
            event_name=EquityOverweightEvent.NAME,
            recguid=recguid,
        )
        self.set_key_fields(
            [
                "EquityHolder",
                "TradedShares",
            ]
        )

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityPledgeEvent(BaseEvent):
    NAME = "EquityPledge"
    # TRIGGERS = ['PledgedShares', 'StartDate', 'ReleasedDate', 'Pledgee', 'TotalPledgedShares', 'TotalHoldingRatio', 'EndDate', 'Pledger', 'TotalHoldingShares']
    TRIGGERS = {
        1: ["PledgedShares"],  # importance: 0.8847324923879948
        2: ["PledgedShares", "StartDate"],  # importance: 0.9238799478033928
        3: ["EndDate", "PledgedShares", "StartDate"],  # importance: 0.9408438451500652
        4: [
            "EndDate",
            "PledgedShares",
            "ReleasedDate",
            "StartDate",
        ],  # importance: 0.9499782514136581
        5: [
            "EndDate",
            "PledgedShares",
            "Pledgee",
            "ReleasedDate",
            "StartDate",
        ],  # importance: 0.9543279686820356
        6: [
            "EndDate",
            "PledgedShares",
            "Pledgee",
            "Pledger",
            "ReleasedDate",
            "StartDate",
        ],  # importance: 0.9560678555893867
        7: [
            "EndDate",
            "PledgedShares",
            "Pledgee",
            "Pledger",
            "ReleasedDate",
            "StartDate",
            "TotalPledgedShares",
        ],  # importance: 0.9565028273162245
        8: [
            "EndDate",
            "PledgedShares",
            "Pledgee",
            "Pledger",
            "ReleasedDate",
            "StartDate",
            "TotalHoldingShares",
            "TotalPledgedShares",
        ],  # importance: 0.9565028273162245
        9: [
            "EndDate",
            "PledgedShares",
            "Pledgee",
            "Pledger",
            "ReleasedDate",
            "StartDate",
            "TotalHoldingRatio",
            "TotalHoldingShares",
            "TotalPledgedShares",
        ],  # importance: 0.9565028273162245
    }
    TRIGGERS["all"] = [
        "PledgedShares",
        "StartDate",
        "Pledgee",
        "Pledger",
        "TotalHoldingShares",
        "TotalPledgedShares",
        "TotalHoldingRatio",
        "EndDate",
        "ReleasedDate",
    ]
    FIELDS = [
        "Pledger",
        "PledgedShares",
        "Pledgee",
        "TotalHoldingShares",
        "TotalHoldingRatio",
        "TotalPledgedShares",
        "StartDate",
        "EndDate",
        "ReleasedDate",
    ]

    def __init__(self, recguid=None):
        # super(EquityPledgeEvent, self).__init__(
        super().__init__(
            EquityPledgeEvent.FIELDS, event_name=EquityPledgeEvent.NAME, recguid=recguid
        )
        self.set_key_fields(
            [
                "Pledger",
                "PledgedShares",
                "Pledgee",
            ]
        )

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


common_fields = ["StockCode", "StockAbbr", "CompanyName", "OtherType"]


event_type2event_class = {
    EquityFreezeEvent.NAME: EquityFreezeEvent,
    EquityRepurchaseEvent.NAME: EquityRepurchaseEvent,
    EquityUnderweightEvent.NAME: EquityUnderweightEvent,
    EquityOverweightEvent.NAME: EquityOverweightEvent,
    EquityPledgeEvent.NAME: EquityPledgeEvent,
}


event_type_fields_list = [
    # name, fields, trigger fields, min_fields_num
    (EquityFreezeEvent.NAME, EquityFreezeEvent.FIELDS, EquityFreezeEvent.TRIGGERS, 5),
    (
        EquityRepurchaseEvent.NAME,
        EquityRepurchaseEvent.FIELDS,
        EquityRepurchaseEvent.TRIGGERS,
        4,
    ),
    (
        EquityUnderweightEvent.NAME,
        EquityUnderweightEvent.FIELDS,
        EquityUnderweightEvent.TRIGGERS,
        4,
    ),
    (
        EquityOverweightEvent.NAME,
        EquityOverweightEvent.FIELDS,
        EquityOverweightEvent.TRIGGERS,
        4,
    ),
    (EquityPledgeEvent.NAME, EquityPledgeEvent.FIELDS, EquityPledgeEvent.TRIGGERS, 7),
]
