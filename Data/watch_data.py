import json
import readline


def load_json(filepath):
    data = []
    with open(filepath, "rt", encoding="utf-8") as fin:
        data = json.load(fin)
    return data


def main():
    # filepath = "typed_sample_train_48.json"
    filepath = "typed_train.json"
    # filepath = "typed_test.json"
    data = load_json(filepath)
    guid2data = {}
    for d in data:
        guid2data[d[0]] = d

    orz = ""
    while orz != "orz":
        orz = input("guid >")
        if orz == "orz":
            break
        if orz not in guid2data:
            print("not exist!")
        else:
            format_data = json.dumps(guid2data[orz], indent=2, ensure_ascii=False)
            print(format_data)


if __name__ == "__main__":
    main()
