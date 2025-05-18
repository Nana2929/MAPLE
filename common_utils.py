import json
import os


def load_jsonl(path: os.PathLike):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def get_meta(meta_path: os.PathLike, id_type: str):
    try:
        data = load_jsonl(meta_path)
    except:
        with open(meta_path, "r") as f:
            data = json.load(f)
    return {x[id_type]: x for x in data}
