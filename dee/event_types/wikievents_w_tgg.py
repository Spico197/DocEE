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


class CognitiveIdentifyCategorizeUnspecified(BaseEvent):
    NAME = "Cognitive.IdentifyCategorize.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6833333333333333
        2: ["IdentifiedObject", "Trigger"],  # importance: 0.7666666666666667
        3: ["IdentifiedObject", "Identifier", "Trigger"],  # importance: 0.8
        4: [
            "IdentifiedObject",
            "Identifier",
            "Place",
            "Trigger",
        ],  # importance: 0.8166666666666667
        5: [
            "IdentifiedObject",
            "IdentifiedRole",
            "Identifier",
            "Place",
            "Trigger",
        ],  # importance: 0.8333333333333334
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Identifier",
        "IdentifiedObject",
        "IdentifiedRole",
        "Place",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class CognitiveInspectionSensoryObserve(BaseEvent):
    NAME = "Cognitive.Inspection.SensoryObserve"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.8648648648648649
        2: ["Observer", "Trigger"],  # importance: 0.918918918918919
        3: ["Instrument", "Observer", "Trigger"],  # importance: 0.9459459459459459
        4: [
            "Instrument",
            "ObservedEntity",
            "Observer",
            "Trigger",
        ],  # importance: 0.9459459459459459
        5: [
            "Instrument",
            "ObservedEntity",
            "Observer",
            "Place",
            "Trigger",
        ],  # importance: 0.9459459459459459
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Observer",
        "ObservedEntity",
        "Place",
        "Instrument",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ConflictAttackUnspecified(BaseEvent):
    NAME = "Conflict.Attack.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.40145985401459855
        2: ["Target", "Trigger"],  # importance: 0.4768856447688564
        3: ["Place", "Target", "Trigger"],  # importance: 0.5352798053527981
        4: ["Attacker", "Place", "Target", "Trigger"],  # importance: 0.5669099756690997
        5: [
            "Attacker",
            "Instrument",
            "Place",
            "Target",
            "Trigger",
        ],  # importance: 0.583941605839416
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Target",
        "Attacker",
        "Place",
        "Instrument",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class LifeInjureUnspecified(BaseEvent):
    NAME = "Life.Injure.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.627906976744186
        2: ["Trigger", "Victim"],  # importance: 0.8106312292358804
        3: ["Instrument", "Trigger", "Victim"],  # importance: 0.8172757475083057
        4: [
            "Injurer",
            "Instrument",
            "Trigger",
            "Victim",
        ],  # importance: 0.8205980066445183
        5: [
            "BodyPart",
            "Injurer",
            "Instrument",
            "Trigger",
            "Victim",
        ],  # importance: 0.8205980066445183
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Victim",
        "Injurer",
        "Instrument",
        "BodyPart",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ConflictAttackDetonateExplode(BaseEvent):
    NAME = "Conflict.Attack.DetonateExplode"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.45443499392466585
        2: ["Place", "Trigger"],  # importance: 0.5370595382746051
        3: ["Place", "Target", "Trigger"],  # importance: 0.5965978128797084
        4: ["Attacker", "Place", "Target", "Trigger"],  # importance: 0.6196840826245443
        5: [
            "Attacker",
            "ExplosiveDevice",
            "Place",
            "Target",
            "Trigger",
        ],  # importance: 0.6281895504252734
        6: [
            "Attacker",
            "ExplosiveDevice",
            "Instrument",
            "Place",
            "Target",
            "Trigger",
        ],  # importance: 0.6318347509113001
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Place",
        "Target",
        "ExplosiveDevice",
        "Attacker",
        "Instrument",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class JusticeChargeIndictUnspecified(BaseEvent):
    NAME = "Justice.ChargeIndict.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6346153846153846
        2: ["Defendant", "Trigger"],  # importance: 0.7692307692307693
        3: ["Defendant", "Place", "Trigger"],  # importance: 0.7884615384615384
        4: [
            "Defendant",
            "Place",
            "Prosecutor",
            "Trigger",
        ],  # importance: 0.8076923076923077
        5: [
            "Defendant",
            "JudgeCourt",
            "Place",
            "Prosecutor",
            "Trigger",
        ],  # importance: 0.8173076923076923
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Defendant",
        "Prosecutor",
        "JudgeCourt",
        "Place",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class JusticeArrestJailDetainUnspecified(BaseEvent):
    NAME = "Justice.ArrestJailDetain.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6223404255319149
        2: ["Detainee", "Trigger"],  # importance: 0.7606382978723404
        3: ["Detainee", "Jailer", "Trigger"],  # importance: 0.7872340425531915
        4: ["Detainee", "Jailer", "Place", "Trigger"],  # importance: 0.8031914893617021
    }
    TRIGGERS["all"] = ["Trigger", "Detainee", "Jailer", "Place"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class JusticeConvictUnspecified(BaseEvent):
    NAME = "Justice.Convict.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6923076923076923
        2: ["Defendant", "Trigger"],  # importance: 0.7884615384615384
        3: ["Defendant", "JudgeCourt", "Trigger"],  # importance: 0.8076923076923077
    }
    TRIGGERS["all"] = ["Trigger", "Defendant", "JudgeCourt"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class JusticeInvestigateCrimeUnspecified(BaseEvent):
    NAME = "Justice.InvestigateCrime.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.7065217391304348
        2: ["Investigator", "Trigger"],  # importance: 0.8260869565217391
        3: ["Defendant", "Investigator", "Trigger"],  # importance: 0.8369565217391305
        4: [
            "Defendant",
            "Investigator",
            "Place",
            "Trigger",
        ],  # importance: 0.8369565217391305
        5: [
            "Defendant",
            "Investigator",
            "Observer",
            "Place",
            "Trigger",
        ],  # importance: 0.8369565217391305
        6: [
            "Defendant",
            "Investigator",
            "ObservedEntity",
            "Observer",
            "Place",
            "Trigger",
        ],  # importance: 0.8369565217391305
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Investigator",
        "Defendant",
        "Place",
        "Observer",
        "ObservedEntity",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ContactContactUnspecified(BaseEvent):
    NAME = "Contact.Contact.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.4405940594059406
        2: ["Participant", "Trigger"],  # importance: 0.7178217821782178
        3: ["Participant", "Topic", "Trigger"],  # importance: 0.7326732673267327
        4: [
            "Participant",
            "Place",
            "Topic",
            "Trigger",
        ],  # importance: 0.7326732673267327
    }
    TRIGGERS["all"] = ["Participant", "Trigger", "Topic", "Place"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class LifeDieUnspecified(BaseEvent):
    NAME = "Life.Die.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.41904761904761906
        2: ["Trigger", "Victim"],  # importance: 0.6628571428571428
        3: ["Killer", "Trigger", "Victim"],  # importance: 0.6819047619047619
        4: ["Killer", "Place", "Trigger", "Victim"],  # importance: 0.6952380952380952
    }
    TRIGGERS["all"] = ["Victim", "Trigger", "Killer", "Place"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ArtifactExistenceDamageDestroyDisableDismantleDamage(BaseEvent):
    NAME = "ArtifactExistence.DamageDestroyDisableDismantle.Damage"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.5208333333333334
        2: ["Artifact", "Trigger"],  # importance: 0.7291666666666666
        3: ["Artifact", "Place", "Trigger"],  # importance: 0.75
        4: ["Artifact", "Instrument", "Place", "Trigger"],  # importance: 0.75
        5: [
            "Artifact",
            "Damager",
            "Instrument",
            "Place",
            "Trigger",
        ],  # importance: 0.75
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Artifact",
        "Place",
        "Instrument",
        "Damager",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ControlImpedeInterfereWithUnspecified(BaseEvent):
    NAME = "Control.ImpedeInterfereWith.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["Impeder", "Trigger"],  # importance: 1.0
        3: ["Impeder", "Place", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Trigger", "Impeder", "Place"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class GenericCrimeGenericCrimeGenericCrime(BaseEvent):
    NAME = "GenericCrime.GenericCrime.GenericCrime"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6533333333333333
        2: ["Perpetrator", "Trigger"],  # importance: 0.7066666666666667
        3: ["Perpetrator", "Trigger", "Victim"],  # importance: 0.7333333333333333
        4: [
            "Perpetrator",
            "Place",
            "Trigger",
            "Victim",
        ],  # importance: 0.7466666666666667
    }
    TRIGGERS["all"] = ["Trigger", "Perpetrator", "Victim", "Place"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class MovementTransportationUnspecified(BaseEvent):
    NAME = "Movement.Transportation.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6666666666666666
        2: ["PassengerArtifact", "Trigger"],  # importance: 0.782051282051282
        3: [
            "Destination",
            "PassengerArtifact",
            "Trigger",
        ],  # importance: 0.8397435897435898
        4: [
            "Destination",
            "PassengerArtifact",
            "Trigger",
            "Vehicle",
        ],  # importance: 0.8461538461538461
        5: [
            "Destination",
            "PassengerArtifact",
            "Transporter",
            "Trigger",
            "Vehicle",
        ],  # importance: 0.8525641025641025
        6: [
            "Destination",
            "Origin",
            "PassengerArtifact",
            "Transporter",
            "Trigger",
            "Vehicle",
        ],  # importance: 0.8525641025641025
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Destination",
        "PassengerArtifact",
        "Transporter",
        "Vehicle",
        "Origin",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ContactContactBroadcast(BaseEvent):
    NAME = "Contact.Contact.Broadcast"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.4153846153846154
        2: ["Communicator", "Trigger"],  # importance: 0.6871794871794872
        3: ["Communicator", "Topic", "Trigger"],  # importance: 0.6923076923076923
        4: [
            "Communicator",
            "Place",
            "Topic",
            "Trigger",
        ],  # importance: 0.6923076923076923
        5: [
            "Communicator",
            "Place",
            "Recipient",
            "Topic",
            "Trigger",
        ],  # importance: 0.6923076923076923
        6: [
            "Communicator",
            "Instrument",
            "Place",
            "Recipient",
            "Topic",
            "Trigger",
        ],  # importance: 0.6923076923076923
    }
    TRIGGERS["all"] = [
        "Communicator",
        "Trigger",
        "Topic",
        "Recipient",
        "Place",
        "Instrument",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ArtifactExistenceDamageDestroyDisableDismantleDestroy(BaseEvent):
    NAME = "ArtifactExistence.DamageDestroyDisableDismantle.Destroy"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.7058823529411765
        2: ["Artifact", "Trigger"],  # importance: 0.8627450980392157
        3: ["Artifact", "Destroyer", "Trigger"],  # importance: 0.8823529411764706
        4: [
            "Artifact",
            "Destroyer",
            "Instrument",
            "Trigger",
        ],  # importance: 0.8823529411764706
        5: [
            "Artifact",
            "Destroyer",
            "Instrument",
            "Place",
            "Trigger",
        ],  # importance: 0.8823529411764706
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Artifact",
        "Destroyer",
        "Place",
        "Instrument",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class MedicalInterventionUnspecified(BaseEvent):
    NAME = "Medical.Intervention.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.8260869565217391
        2: ["Patient", "Trigger"],  # importance: 1.0
        3: ["Patient", "Treater", "Trigger"],  # importance: 1.0
        4: ["Patient", "Place", "Treater", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Trigger", "Patient", "Treater", "Place"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ConflictDemonstrateDemonstrateWithViolence(BaseEvent):
    NAME = "Conflict.Demonstrate.DemonstrateWithViolence"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["Regulator", "Trigger"],  # importance: 1.0
        3: ["Demonstrator", "Regulator", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Trigger", "Demonstrator", "Regulator"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ConflictDemonstrateUnspecified(BaseEvent):
    NAME = "Conflict.Demonstrate.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.5625
        2: ["Target", "Trigger"],  # importance: 0.625
        3: ["Target", "Topic", "Trigger"],  # importance: 0.65625
        4: ["Demonstrator", "Target", "Topic", "Trigger"],  # importance: 0.65625
    }
    TRIGGERS["all"] = ["Trigger", "Demonstrator", "Target", "Topic"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ContactThreatenCoerceUnspecified(BaseEvent):
    NAME = "Contact.ThreatenCoerce.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6153846153846154
        2: ["Recipient", "Trigger"],  # importance: 0.7692307692307693
        3: ["Communicator", "Recipient", "Trigger"],  # importance: 0.7692307692307693
    }
    TRIGGERS["all"] = ["Trigger", "Recipient", "Communicator"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ContactRequestCommandBroadcast(BaseEvent):
    NAME = "Contact.RequestCommand.Broadcast"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["Recipient", "Trigger"],  # importance: 1.0
        3: ["Communicator", "Recipient", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Trigger", "Communicator", "Recipient"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ContactContactMeet(BaseEvent):
    NAME = "Contact.Contact.Meet"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.660377358490566
        2: ["Participant", "Trigger"],  # importance: 0.7547169811320755
        3: ["Participant", "Place", "Trigger"],  # importance: 0.7924528301886793
        4: [
            "Participant",
            "Place",
            "Topic",
            "Trigger",
        ],  # importance: 0.7924528301886793
    }
    TRIGGERS["all"] = ["Trigger", "Participant", "Place", "Topic"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class MovementTransportationEvacuation(BaseEvent):
    NAME = "Movement.Transportation.Evacuation"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["PassengerArtifact", "Trigger"],  # importance: 1.0
        3: ["PassengerArtifact", "Transporter", "Trigger"],  # importance: 1.0
        4: ["Origin", "PassengerArtifact", "Transporter", "Trigger"],  # importance: 1.0
        5: [
            "Destination",
            "Origin",
            "PassengerArtifact",
            "Transporter",
            "Trigger",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = [
        "Trigger",
        "PassengerArtifact",
        "Origin",
        "Transporter",
        "Destination",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class JusticeAcquitUnspecified(BaseEvent):
    NAME = "Justice.Acquit.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["Defendant", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Trigger", "Defendant"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ArtifactExistenceManufactureAssembleUnspecified(BaseEvent):
    NAME = "ArtifactExistence.ManufactureAssemble.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.8115942028985508
        2: ["Components", "Trigger"],  # importance: 0.9130434782608695
        3: ["Artifact", "Components", "Trigger"],  # importance: 0.9710144927536232
        4: [
            "Artifact",
            "Components",
            "Place",
            "Trigger",
        ],  # importance: 0.9710144927536232
        5: [
            "Artifact",
            "Components",
            "ManufacturerAssembler",
            "Place",
            "Trigger",
        ],  # importance: 0.9710144927536232
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Artifact",
        "Components",
        "ManufacturerAssembler",
        "Place",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ArtifactExistenceDamageDestroyDisableDismantleDismantle(BaseEvent):
    NAME = "ArtifactExistence.DamageDestroyDisableDismantle.Dismantle"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.75
        2: ["Instrument", "Trigger"],  # importance: 0.875
        3: ["Dismantler", "Instrument", "Trigger"],  # importance: 1.0
        4: ["Dismantler", "Instrument", "Place", "Trigger"],  # importance: 1.0
        5: [
            "Components",
            "Dismantler",
            "Instrument",
            "Place",
            "Trigger",
        ],  # importance: 1.0
        6: [
            "Artifact",
            "Components",
            "Dismantler",
            "Instrument",
            "Place",
            "Trigger",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Artifact",
        "Instrument",
        "Components",
        "Place",
        "Dismantler",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class JusticeSentenceUnspecified(BaseEvent):
    NAME = "Justice.Sentence.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6885245901639344
        2: ["JudgeCourt", "Trigger"],  # importance: 0.7704918032786885
        3: ["Defendant", "JudgeCourt", "Trigger"],  # importance: 0.8032786885245902
        4: [
            "Defendant",
            "JudgeCourt",
            "Place",
            "Trigger",
        ],  # importance: 0.8032786885245902
    }
    TRIGGERS["all"] = ["Trigger", "Defendant", "JudgeCourt", "Place"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class JusticeTrialHearingUnspecified(BaseEvent):
    NAME = "Justice.TrialHearing.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.5365853658536586
        2: ["JudgeCourt", "Trigger"],  # importance: 0.6097560975609756
        3: ["Defendant", "JudgeCourt", "Trigger"],  # importance: 0.6829268292682927
        4: [
            "Defendant",
            "JudgeCourt",
            "Place",
            "Trigger",
        ],  # importance: 0.7073170731707317
        5: [
            "Defendant",
            "JudgeCourt",
            "Place",
            "Prosecutor",
            "Trigger",
        ],  # importance: 0.7073170731707317
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Defendant",
        "JudgeCourt",
        "Place",
        "Prosecutor",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class TransactionExchangeBuySellUnspecified(BaseEvent):
    NAME = "Transaction.ExchangeBuySell.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6811594202898551
        2: ["AcquiredEntity", "Trigger"],  # importance: 0.8405797101449275
        3: ["AcquiredEntity", "Giver", "Trigger"],  # importance: 0.8985507246376812
        4: [
            "AcquiredEntity",
            "Giver",
            "Recipient",
            "Trigger",
        ],  # importance: 0.927536231884058
        5: [
            "AcquiredEntity",
            "Giver",
            "PaymentBarter",
            "Recipient",
            "Trigger",
        ],  # importance: 0.927536231884058
    }
    TRIGGERS["all"] = [
        "Trigger",
        "AcquiredEntity",
        "Giver",
        "Recipient",
        "PaymentBarter",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class MovementTransportationPreventPassage(BaseEvent):
    NAME = "Movement.Transportation.PreventPassage"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.45454545454545453
        2: ["Trigger", "Vehicle"],  # importance: 0.5454545454545454
        3: ["Preventer", "Trigger", "Vehicle"],  # importance: 0.5454545454545454
        4: [
            "Preventer",
            "Transporter",
            "Trigger",
            "Vehicle",
        ],  # importance: 0.5454545454545454
        5: [
            "Destination",
            "Preventer",
            "Transporter",
            "Trigger",
            "Vehicle",
        ],  # importance: 0.5454545454545454
        6: [
            "Destination",
            "PassengerArtifact",
            "Preventer",
            "Transporter",
            "Trigger",
            "Vehicle",
        ],  # importance: 0.5454545454545454
        7: [
            "Destination",
            "Origin",
            "PassengerArtifact",
            "Preventer",
            "Transporter",
            "Trigger",
            "Vehicle",
        ],  # importance: 0.5454545454545454
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Destination",
        "Preventer",
        "Vehicle",
        "Transporter",
        "Origin",
        "PassengerArtifact",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ContactContactCorrespondence(BaseEvent):
    NAME = "Contact.Contact.Correspondence"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.7777777777777778
        2: ["Participant", "Trigger"],  # importance: 0.9444444444444444
        3: ["Participant", "Topic", "Trigger"],  # importance: 0.9444444444444444
        4: [
            "Participant",
            "Place",
            "Topic",
            "Trigger",
        ],  # importance: 0.9444444444444444
    }
    TRIGGERS["all"] = ["Participant", "Trigger", "Topic", "Place"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ContactThreatenCoerceBroadcast(BaseEvent):
    NAME = "Contact.ThreatenCoerce.Broadcast"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["Recipient", "Trigger"],  # importance: 1.0
        3: ["Communicator", "Recipient", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Trigger", "Recipient", "Communicator"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ContactRequestCommandUnspecified(BaseEvent):
    NAME = "Contact.RequestCommand.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.75
        2: ["Communicator", "Trigger"],  # importance: 0.84375
        3: ["Communicator", "Place", "Trigger"],  # importance: 0.875
        4: ["Communicator", "Place", "Recipient", "Trigger"],  # importance: 0.875
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Recipient",
        "Communicator",
        "Place",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ConflictDefeatUnspecified(BaseEvent):
    NAME = "Conflict.Defeat.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["Trigger", "Victor"],  # importance: 1.0
        3: ["Defeated", "Trigger", "Victor"],  # importance: 1.0
        4: ["Defeated", "Place", "Trigger", "Victor"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Victor", "Trigger", "Defeated", "Place"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class LifeInfectUnspecified(BaseEvent):
    NAME = "Life.Infect.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["Trigger", "Victim"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Trigger", "Victim"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class CognitiveResearchUnspecified(BaseEvent):
    NAME = "Cognitive.Research.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.8181818181818182
        2: ["Researcher", "Trigger"],  # importance: 1.0
        3: ["Place", "Researcher", "Trigger"],  # importance: 1.0
        4: ["Place", "Researcher", "Subject", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Trigger", "Subject", "Researcher", "Place"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class DisasterCrashUnspecified(BaseEvent):
    NAME = "Disaster.Crash.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.5
        2: ["Place", "Trigger"],  # importance: 0.7
        3: ["CrashObject", "Place", "Trigger"],  # importance: 0.8
        4: ["CrashObject", "Place", "Trigger", "Vehicle"],  # importance: 0.8
    }
    TRIGGERS["all"] = ["Trigger", "CrashObject", "Place", "Vehicle"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ArtifactExistenceDamageDestroyDisableDismantleUnspecified(BaseEvent):
    NAME = "ArtifactExistence.DamageDestroyDisableDismantle.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.36363636363636365
        2: ["Artifact", "Trigger"],  # importance: 1.0
        3: ["Artifact", "Instrument", "Trigger"],  # importance: 1.0
        4: ["Artifact", "DamagerDestroyer", "Instrument", "Trigger"],  # importance: 1.0
        5: [
            "Artifact",
            "DamagerDestroyer",
            "Instrument",
            "Place",
            "Trigger",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = [
        "Artifact",
        "Trigger",
        "DamagerDestroyer",
        "Place",
        "Instrument",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class MovementTransportationIllegalTransportation(BaseEvent):
    NAME = "Movement.Transportation.IllegalTransportation"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.5714285714285714
        2: ["Destination", "Trigger"],  # importance: 0.7857142857142857
        3: [
            "Destination",
            "PassengerArtifact",
            "Trigger",
        ],  # importance: 0.8571428571428571
        4: [
            "Destination",
            "PassengerArtifact",
            "Transporter",
            "Trigger",
        ],  # importance: 0.9285714285714286
        5: [
            "Destination",
            "PassengerArtifact",
            "Transporter",
            "Trigger",
            "Vehicle",
        ],  # importance: 0.9285714285714286
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Destination",
        "PassengerArtifact",
        "Vehicle",
        "Transporter",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ContactThreatenCoerceCorrespondence(BaseEvent):
    NAME = "Contact.ThreatenCoerce.Correspondence"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.3333333333333333
        2: ["Communicator", "Trigger"],  # importance: 1.0
        3: ["Communicator", "Recipient", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Communicator", "Trigger", "Recipient"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class PersonnelEndPositionUnspecified(BaseEvent):
    NAME = "Personnel.EndPosition.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["Employee", "Trigger"],  # importance: 1.0
        3: ["Employee", "PlaceOfEmployment", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Trigger", "Employee", "PlaceOfEmployment"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ArtifactExistenceDamageDestroyDisableDismantleDisableDefuse(BaseEvent):
    NAME = "ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["Instrument", "Trigger"],  # importance: 1.0
        3: ["Disabler", "Instrument", "Trigger"],  # importance: 1.0
        4: ["Artifact", "Disabler", "Instrument", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Artifact",
        "Disabler",
        "Instrument",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class PersonnelStartPositionUnspecified(BaseEvent):
    NAME = "Personnel.StartPosition.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["Position", "Trigger"],  # importance: 1.0
        3: ["Employee", "Position", "Trigger"],  # importance: 1.0
        4: ["Employee", "Place", "Position", "Trigger"],  # importance: 1.0
        5: [
            "Employee",
            "Place",
            "PlaceOfEmployment",
            "Position",
            "Trigger",
        ],  # importance: 1.0
    }
    TRIGGERS["all"] = [
        "Trigger",
        "Employee",
        "Position",
        "Place",
        "PlaceOfEmployment",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class CognitiveTeachingTrainingLearningUnspecified(BaseEvent):
    NAME = "Cognitive.TeachingTrainingLearning.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.8571428571428571
        2: ["Learner", "Trigger"],  # importance: 1.0
        3: ["Learner", "TeacherTrainer", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Trigger", "Learner", "TeacherTrainer"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class JusticeReleaseParoleUnspecified(BaseEvent):
    NAME = "Justice.ReleaseParole.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.7333333333333333
        2: ["Defendant", "Trigger"],  # importance: 0.8666666666666667
        3: ["Defendant", "JudgeCourt", "Trigger"],  # importance: 0.8666666666666667
    }
    TRIGGERS["all"] = ["Trigger", "Defendant", "JudgeCourt"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class TransactionDonationUnspecified(BaseEvent):
    NAME = "Transaction.Donation.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["ArtifactMoney", "Trigger"],  # importance: 1.0
        3: ["ArtifactMoney", "Giver", "Trigger"],  # importance: 1.0
        4: ["ArtifactMoney", "Giver", "Recipient", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = [
        "ArtifactMoney",
        "Recipient",
        "Trigger",
        "Giver",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class DisasterDiseaseOutbreakUnspecified(BaseEvent):
    NAME = "Disaster.DiseaseOutbreak.Unspecified"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6
        2: ["Place", "Trigger"],  # importance: 0.7333333333333333
        3: ["Place", "Trigger", "Victim"],  # importance: 0.7333333333333333
        4: ["Disease", "Place", "Trigger", "Victim"],  # importance: 0.7333333333333333
    }
    TRIGGERS["all"] = ["Trigger", "Place", "Victim", "Disease"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ContactRequestCommandMeet(BaseEvent):
    NAME = "Contact.RequestCommand.Meet"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 0.6666666666666666
        2: ["Recipient", "Trigger"],  # importance: 1.0
        3: ["Communicator", "Recipient", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = ["Trigger", "Recipient", "Communicator"]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


class ContactRequestCommandCorrespondence(BaseEvent):
    NAME = "Contact.RequestCommand.Correspondence"
    TRIGGERS = {
        1: ["Trigger"],  # importance: 1.0
        2: ["Topic", "Trigger"],  # importance: 1.0
        3: ["Recipient", "Topic", "Trigger"],  # importance: 1.0
        4: ["Communicator", "Recipient", "Topic", "Trigger"],  # importance: 1.0
    }
    TRIGGERS["all"] = [
        "Topic",
        "Recipient",
        "Trigger",
        "Communicator",
    ]
    FIELDS = TRIGGERS["all"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.TRIGGERS)


common_fields = []


event_type2event_class = {
    CognitiveIdentifyCategorizeUnspecified.NAME: CognitiveIdentifyCategorizeUnspecified,
    CognitiveInspectionSensoryObserve.NAME: CognitiveInspectionSensoryObserve,
    ConflictAttackUnspecified.NAME: ConflictAttackUnspecified,
    LifeInjureUnspecified.NAME: LifeInjureUnspecified,
    ConflictAttackDetonateExplode.NAME: ConflictAttackDetonateExplode,
    JusticeChargeIndictUnspecified.NAME: JusticeChargeIndictUnspecified,
    JusticeArrestJailDetainUnspecified.NAME: JusticeArrestJailDetainUnspecified,
    JusticeConvictUnspecified.NAME: JusticeConvictUnspecified,
    JusticeInvestigateCrimeUnspecified.NAME: JusticeInvestigateCrimeUnspecified,
    ContactContactUnspecified.NAME: ContactContactUnspecified,
    LifeDieUnspecified.NAME: LifeDieUnspecified,
    ArtifactExistenceDamageDestroyDisableDismantleDamage.NAME: ArtifactExistenceDamageDestroyDisableDismantleDamage,
    ControlImpedeInterfereWithUnspecified.NAME: ControlImpedeInterfereWithUnspecified,
    GenericCrimeGenericCrimeGenericCrime.NAME: GenericCrimeGenericCrimeGenericCrime,
    MovementTransportationUnspecified.NAME: MovementTransportationUnspecified,
    ContactContactBroadcast.NAME: ContactContactBroadcast,
    ArtifactExistenceDamageDestroyDisableDismantleDestroy.NAME: ArtifactExistenceDamageDestroyDisableDismantleDestroy,
    MedicalInterventionUnspecified.NAME: MedicalInterventionUnspecified,
    ConflictDemonstrateDemonstrateWithViolence.NAME: ConflictDemonstrateDemonstrateWithViolence,
    ConflictDemonstrateUnspecified.NAME: ConflictDemonstrateUnspecified,
    ContactThreatenCoerceUnspecified.NAME: ContactThreatenCoerceUnspecified,
    ContactRequestCommandBroadcast.NAME: ContactRequestCommandBroadcast,
    ContactContactMeet.NAME: ContactContactMeet,
    MovementTransportationEvacuation.NAME: MovementTransportationEvacuation,
    JusticeAcquitUnspecified.NAME: JusticeAcquitUnspecified,
    ArtifactExistenceManufactureAssembleUnspecified.NAME: ArtifactExistenceManufactureAssembleUnspecified,
    ArtifactExistenceDamageDestroyDisableDismantleDismantle.NAME: ArtifactExistenceDamageDestroyDisableDismantleDismantle,
    JusticeSentenceUnspecified.NAME: JusticeSentenceUnspecified,
    JusticeTrialHearingUnspecified.NAME: JusticeTrialHearingUnspecified,
    TransactionExchangeBuySellUnspecified.NAME: TransactionExchangeBuySellUnspecified,
    MovementTransportationPreventPassage.NAME: MovementTransportationPreventPassage,
    ContactContactCorrespondence.NAME: ContactContactCorrespondence,
    ContactThreatenCoerceBroadcast.NAME: ContactThreatenCoerceBroadcast,
    ContactRequestCommandUnspecified.NAME: ContactRequestCommandUnspecified,
    ConflictDefeatUnspecified.NAME: ConflictDefeatUnspecified,
    LifeInfectUnspecified.NAME: LifeInfectUnspecified,
    CognitiveResearchUnspecified.NAME: CognitiveResearchUnspecified,
    DisasterCrashUnspecified.NAME: DisasterCrashUnspecified,
    ArtifactExistenceDamageDestroyDisableDismantleUnspecified.NAME: ArtifactExistenceDamageDestroyDisableDismantleUnspecified,
    MovementTransportationIllegalTransportation.NAME: MovementTransportationIllegalTransportation,
    ContactThreatenCoerceCorrespondence.NAME: ContactThreatenCoerceCorrespondence,
    PersonnelEndPositionUnspecified.NAME: PersonnelEndPositionUnspecified,
    ArtifactExistenceDamageDestroyDisableDismantleDisableDefuse.NAME: ArtifactExistenceDamageDestroyDisableDismantleDisableDefuse,
    PersonnelStartPositionUnspecified.NAME: PersonnelStartPositionUnspecified,
    CognitiveTeachingTrainingLearningUnspecified.NAME: CognitiveTeachingTrainingLearningUnspecified,
    JusticeReleaseParoleUnspecified.NAME: JusticeReleaseParoleUnspecified,
    TransactionDonationUnspecified.NAME: TransactionDonationUnspecified,
    DisasterDiseaseOutbreakUnspecified.NAME: DisasterDiseaseOutbreakUnspecified,
    ContactRequestCommandMeet.NAME: ContactRequestCommandMeet,
    ContactRequestCommandCorrespondence.NAME: ContactRequestCommandCorrespondence,
}


event_type_fields_list = [
    # name, fields, trigger fields, min_fields_num
    (
        CognitiveIdentifyCategorizeUnspecified.NAME,
        CognitiveIdentifyCategorizeUnspecified.FIELDS,
        CognitiveIdentifyCategorizeUnspecified.TRIGGERS,
        1,
    ),
    (
        CognitiveInspectionSensoryObserve.NAME,
        CognitiveInspectionSensoryObserve.FIELDS,
        CognitiveInspectionSensoryObserve.TRIGGERS,
        1,
    ),
    (
        ConflictAttackUnspecified.NAME,
        ConflictAttackUnspecified.FIELDS,
        ConflictAttackUnspecified.TRIGGERS,
        1,
    ),
    (
        LifeInjureUnspecified.NAME,
        LifeInjureUnspecified.FIELDS,
        LifeInjureUnspecified.TRIGGERS,
        1,
    ),
    (
        ConflictAttackDetonateExplode.NAME,
        ConflictAttackDetonateExplode.FIELDS,
        ConflictAttackDetonateExplode.TRIGGERS,
        1,
    ),
    (
        JusticeChargeIndictUnspecified.NAME,
        JusticeChargeIndictUnspecified.FIELDS,
        JusticeChargeIndictUnspecified.TRIGGERS,
        1,
    ),
    (
        JusticeArrestJailDetainUnspecified.NAME,
        JusticeArrestJailDetainUnspecified.FIELDS,
        JusticeArrestJailDetainUnspecified.TRIGGERS,
        1,
    ),
    (
        JusticeConvictUnspecified.NAME,
        JusticeConvictUnspecified.FIELDS,
        JusticeConvictUnspecified.TRIGGERS,
        1,
    ),
    (
        JusticeInvestigateCrimeUnspecified.NAME,
        JusticeInvestigateCrimeUnspecified.FIELDS,
        JusticeInvestigateCrimeUnspecified.TRIGGERS,
        1,
    ),
    (
        ContactContactUnspecified.NAME,
        ContactContactUnspecified.FIELDS,
        ContactContactUnspecified.TRIGGERS,
        1,
    ),
    (
        LifeDieUnspecified.NAME,
        LifeDieUnspecified.FIELDS,
        LifeDieUnspecified.TRIGGERS,
        1,
    ),
    (
        ArtifactExistenceDamageDestroyDisableDismantleDamage.NAME,
        ArtifactExistenceDamageDestroyDisableDismantleDamage.FIELDS,
        ArtifactExistenceDamageDestroyDisableDismantleDamage.TRIGGERS,
        1,
    ),
    (
        ControlImpedeInterfereWithUnspecified.NAME,
        ControlImpedeInterfereWithUnspecified.FIELDS,
        ControlImpedeInterfereWithUnspecified.TRIGGERS,
        1,
    ),
    (
        GenericCrimeGenericCrimeGenericCrime.NAME,
        GenericCrimeGenericCrimeGenericCrime.FIELDS,
        GenericCrimeGenericCrimeGenericCrime.TRIGGERS,
        1,
    ),
    (
        MovementTransportationUnspecified.NAME,
        MovementTransportationUnspecified.FIELDS,
        MovementTransportationUnspecified.TRIGGERS,
        1,
    ),
    (
        ContactContactBroadcast.NAME,
        ContactContactBroadcast.FIELDS,
        ContactContactBroadcast.TRIGGERS,
        1,
    ),
    (
        ArtifactExistenceDamageDestroyDisableDismantleDestroy.NAME,
        ArtifactExistenceDamageDestroyDisableDismantleDestroy.FIELDS,
        ArtifactExistenceDamageDestroyDisableDismantleDestroy.TRIGGERS,
        1,
    ),
    (
        MedicalInterventionUnspecified.NAME,
        MedicalInterventionUnspecified.FIELDS,
        MedicalInterventionUnspecified.TRIGGERS,
        1,
    ),
    (
        ConflictDemonstrateDemonstrateWithViolence.NAME,
        ConflictDemonstrateDemonstrateWithViolence.FIELDS,
        ConflictDemonstrateDemonstrateWithViolence.TRIGGERS,
        1,
    ),
    (
        ConflictDemonstrateUnspecified.NAME,
        ConflictDemonstrateUnspecified.FIELDS,
        ConflictDemonstrateUnspecified.TRIGGERS,
        1,
    ),
    (
        ContactThreatenCoerceUnspecified.NAME,
        ContactThreatenCoerceUnspecified.FIELDS,
        ContactThreatenCoerceUnspecified.TRIGGERS,
        1,
    ),
    (
        ContactRequestCommandBroadcast.NAME,
        ContactRequestCommandBroadcast.FIELDS,
        ContactRequestCommandBroadcast.TRIGGERS,
        1,
    ),
    (
        ContactContactMeet.NAME,
        ContactContactMeet.FIELDS,
        ContactContactMeet.TRIGGERS,
        1,
    ),
    (
        MovementTransportationEvacuation.NAME,
        MovementTransportationEvacuation.FIELDS,
        MovementTransportationEvacuation.TRIGGERS,
        1,
    ),
    (
        JusticeAcquitUnspecified.NAME,
        JusticeAcquitUnspecified.FIELDS,
        JusticeAcquitUnspecified.TRIGGERS,
        1,
    ),
    (
        ArtifactExistenceManufactureAssembleUnspecified.NAME,
        ArtifactExistenceManufactureAssembleUnspecified.FIELDS,
        ArtifactExistenceManufactureAssembleUnspecified.TRIGGERS,
        1,
    ),
    (
        ArtifactExistenceDamageDestroyDisableDismantleDismantle.NAME,
        ArtifactExistenceDamageDestroyDisableDismantleDismantle.FIELDS,
        ArtifactExistenceDamageDestroyDisableDismantleDismantle.TRIGGERS,
        1,
    ),
    (
        JusticeSentenceUnspecified.NAME,
        JusticeSentenceUnspecified.FIELDS,
        JusticeSentenceUnspecified.TRIGGERS,
        1,
    ),
    (
        JusticeTrialHearingUnspecified.NAME,
        JusticeTrialHearingUnspecified.FIELDS,
        JusticeTrialHearingUnspecified.TRIGGERS,
        1,
    ),
    (
        TransactionExchangeBuySellUnspecified.NAME,
        TransactionExchangeBuySellUnspecified.FIELDS,
        TransactionExchangeBuySellUnspecified.TRIGGERS,
        1,
    ),
    (
        MovementTransportationPreventPassage.NAME,
        MovementTransportationPreventPassage.FIELDS,
        MovementTransportationPreventPassage.TRIGGERS,
        1,
    ),
    (
        ContactContactCorrespondence.NAME,
        ContactContactCorrespondence.FIELDS,
        ContactContactCorrespondence.TRIGGERS,
        1,
    ),
    (
        ContactThreatenCoerceBroadcast.NAME,
        ContactThreatenCoerceBroadcast.FIELDS,
        ContactThreatenCoerceBroadcast.TRIGGERS,
        1,
    ),
    (
        ContactRequestCommandUnspecified.NAME,
        ContactRequestCommandUnspecified.FIELDS,
        ContactRequestCommandUnspecified.TRIGGERS,
        1,
    ),
    (
        ConflictDefeatUnspecified.NAME,
        ConflictDefeatUnspecified.FIELDS,
        ConflictDefeatUnspecified.TRIGGERS,
        1,
    ),
    (
        LifeInfectUnspecified.NAME,
        LifeInfectUnspecified.FIELDS,
        LifeInfectUnspecified.TRIGGERS,
        1,
    ),
    (
        CognitiveResearchUnspecified.NAME,
        CognitiveResearchUnspecified.FIELDS,
        CognitiveResearchUnspecified.TRIGGERS,
        1,
    ),
    (
        DisasterCrashUnspecified.NAME,
        DisasterCrashUnspecified.FIELDS,
        DisasterCrashUnspecified.TRIGGERS,
        1,
    ),
    (
        ArtifactExistenceDamageDestroyDisableDismantleUnspecified.NAME,
        ArtifactExistenceDamageDestroyDisableDismantleUnspecified.FIELDS,
        ArtifactExistenceDamageDestroyDisableDismantleUnspecified.TRIGGERS,
        1,
    ),
    (
        MovementTransportationIllegalTransportation.NAME,
        MovementTransportationIllegalTransportation.FIELDS,
        MovementTransportationIllegalTransportation.TRIGGERS,
        1,
    ),
    (
        ContactThreatenCoerceCorrespondence.NAME,
        ContactThreatenCoerceCorrespondence.FIELDS,
        ContactThreatenCoerceCorrespondence.TRIGGERS,
        1,
    ),
    (
        PersonnelEndPositionUnspecified.NAME,
        PersonnelEndPositionUnspecified.FIELDS,
        PersonnelEndPositionUnspecified.TRIGGERS,
        1,
    ),
    (
        ArtifactExistenceDamageDestroyDisableDismantleDisableDefuse.NAME,
        ArtifactExistenceDamageDestroyDisableDismantleDisableDefuse.FIELDS,
        ArtifactExistenceDamageDestroyDisableDismantleDisableDefuse.TRIGGERS,
        1,
    ),
    (
        PersonnelStartPositionUnspecified.NAME,
        PersonnelStartPositionUnspecified.FIELDS,
        PersonnelStartPositionUnspecified.TRIGGERS,
        1,
    ),
    (
        CognitiveTeachingTrainingLearningUnspecified.NAME,
        CognitiveTeachingTrainingLearningUnspecified.FIELDS,
        CognitiveTeachingTrainingLearningUnspecified.TRIGGERS,
        1,
    ),
    (
        JusticeReleaseParoleUnspecified.NAME,
        JusticeReleaseParoleUnspecified.FIELDS,
        JusticeReleaseParoleUnspecified.TRIGGERS,
        1,
    ),
    (
        TransactionDonationUnspecified.NAME,
        TransactionDonationUnspecified.FIELDS,
        TransactionDonationUnspecified.TRIGGERS,
        1,
    ),
    (
        DisasterDiseaseOutbreakUnspecified.NAME,
        DisasterDiseaseOutbreakUnspecified.FIELDS,
        DisasterDiseaseOutbreakUnspecified.TRIGGERS,
        1,
    ),
    (
        ContactRequestCommandMeet.NAME,
        ContactRequestCommandMeet.FIELDS,
        ContactRequestCommandMeet.TRIGGERS,
        1,
    ),
    (
        ContactRequestCommandCorrespondence.NAME,
        ContactRequestCommandCorrespondence.FIELDS,
        ContactRequestCommandCorrespondence.TRIGGERS,
        1,
    ),
]
