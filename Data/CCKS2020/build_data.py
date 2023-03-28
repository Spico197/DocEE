from dee.event_types import get_doc_type
from dee.utils import default_dump_json, default_load_json

if __name__ == "__main__":
    for dname in ["train", "dev", "test"]:
        data = default_load_json(f"Data/CCKS2020/{dname}.json")
        new_data = []
        for doc_id, content in data:
            doc_type = get_doc_type(content["recguid_eventname_eventdict_list"])
            content["doc_type"] = doc_type
            new_data.append([doc_id, content])
        default_dump_json(new_data, f"Data/CCKS2020/typed_{dname}.json")
