import json


def load_json(filepath):
    with open(filepath, "rt", encoding="utf-8") as fin:
        data = json.load(fin)
        return data


def convert2combination(data):
    spans = data["ann_valid_mspans"]
    num_spans = len(spans)
    span2idx = {span: span_idx for span_idx, span in enumerate(spans)}
    combination = []
    for instance in data["recguid_eventname_eventdict_list"]:
        event = instance[2]
        span_idxs = [
            span2idx[x] for x in filter(lambda x: x is not None, set(event.values()))
        ]
        span_idxs.sort()
        spans_in_event = tuple(span_idxs)
        combination.append(spans_in_event)
    return combination, num_spans


if __name__ == "__main__":
    data = load_json("train_most_sophiscated.json")
    combination, num_spans = convert2combination(data)
    print(num_spans, combination)
