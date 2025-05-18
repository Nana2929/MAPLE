## Saving yelp'23 to smaller files
import json
from pathlib import Path

SIZE = 5000

# =========== utilities =============
def save_jsonl(data: dict, save_path: str):
    with open(save_path, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')
def read_jsonl(file_path: str):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data



root_dir = Path("../yelp_2023/yelp_2023_pruned")
save_dir = root_dir / "partitions"
save_dir.mkdir(exist_ok=True)
input_data_path = root_dir / "yelp_academic_dataset_review_pruned.json"
input_data = read_jsonl(input_data_path)


output_data = []
for i in range(len(input_data)):
    output_data.append(input_data[i])
    if i % SIZE == 0:
        if i > 0:
            print(f"Saving {i} with {len(output_data)} entries")
            save_jsonl(output_data, save_dir / f"{i}.jsonl")
            output_data = []

if len(output_data) > 0:
    print(f"Saving {i} with {len(output_data)} entries")
    save_jsonl(output_data, save_dir / f"{i}.jsonl")
    output_data = []




