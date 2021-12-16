import json
from collections import defaultdict


def load_json(filepath):
    data = []
    with open(filepath, "rt", encoding="utf-8") as fin:
        data = json.load(fin)
    return data


def dump_json(obj, filepath, **kwargs):
    with open(filepath, "wt", encoding="utf-8") as fout:
        json.dump(obj, fout, ensure_ascii=False, **kwargs)


def get_ent_type_info(filepath, dump_filepath):
    type2ents = defaultdict(list)
    data = load_json(filepath)
    for doc_id, doc_content in data:
        ent2field = doc_content["ann_mspan2guess_field"]
        for ent, field in ent2field.items():
            type2ents[field].append(ent)
    print(f"#types: {len(type2ents)}, types: {type2ents.keys()}")
    dump_json(type2ents, dump_filepath, indent=2)


if __name__ == "__main__":
    get_ent_type_info("dev.json", "dev_field2ents.json")
