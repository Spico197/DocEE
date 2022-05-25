import importlib
import os

__current_dir = os.listdir(os.path.dirname(__file__))
AVAILABLE_TEMPLATES = list(
    map(
        lambda x: x.split(".")[0],
        filter(lambda x: x.endswith(".py") and x != "__init__.py", __current_dir),
    )
)


def get_event_template(template_name):
    assert template_name in AVAILABLE_TEMPLATES
    template = importlib.import_module(f".{template_name}", "dee.event_types")
    return template


if __name__ == "__main__":
    template = get_event_template("zheng2019")
    print(template)
