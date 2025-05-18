import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def save_index(path: str, index: list):
    with open(path, 'w') as f:
        for i, idx in enumerate(index):
            if i == len(index) - 1:
                f.write(f"{idx}")
            else:
                f.write(f"{idx} ")
    return

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def read_jsonl_file(file_path: str):
    return [json.loads(line) for line in open(file_path)]


def write_jsonl_file(
    data: dict,
    out_filename: str,
):
    with open(out_filename, "w") as output:
        for record in data:
            output.write(f"{json.dumps(record)}\n")
    return


def read_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return


def save_csv(data: list[str], out_filename: str):
    # pandas
    df = pd.DataFrame(data)
    df.to_csv(out_filename, index=False, header=False)



def read_jsonl_file(path):
    with open(path) as f:
        data = [json.loads(line) for line in f]
    return data

def load_uie_files(root: str, split_name: str=None):
    if split_name is None:
        dir_path = Path(root)
    else: dir_path = Path(root) / (split_name + "_uie")

    print("loading from", dir_path)
    all_files = list(dir_path.glob("*.json"))
    data = []
    if len(all_files) == 0:
        print("no files found")
    for file in all_files:
        data.extend(read_jsonl_file(file))
    return data