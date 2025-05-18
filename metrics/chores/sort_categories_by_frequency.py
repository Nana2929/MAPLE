from collections import defaultdict
import pandas as pd
import pickle
from tqdm import tqdm
import fire

dataset_dir = {
    "yelp": "nete_format_data/yelp",
    "yelp23": "nete_format_data/yelp23",
}


def load_index(index_path):
    with open(index_path, "r") as f:
        index = [int(x) for x in f.readline().split(" ")]
    return index


def get_aspect_count(dataset, fold=1):
    aspect_count = defaultdict(int)

    aspect_category_index_path = dataset_dir[dataset] + "/aspect_category_index.csv"
    df = pd.read_csv(aspect_category_index_path)

    print(f"Loading {dataset} train index")
    train_index = load_index(f"{dataset_dir[dataset]}/{fold}/train.index")
    train_index = set(train_index)

    reverse_df = {cat: index for index, cat in enumerate(df["category"])}
    print(f"Calculating aspect count...")
    for idx, review in enumerate(tqdm(reviews)):
        if idx not in train_index:
            continue
        category_name = review["category"]
        index = reverse_df[category_name]
        aspect_count[index] += 1
    aspect_count = sorted(aspect_count.items(), key=lambda x: x[1], reverse=True)
    ret = []
    for index, count in aspect_count:
        cat = df.iloc[index]["category"]
        ret.append((index, cat, count))
    return ret


if __name__ == "__main__":
    import sys

    for arg in sys.argv:
        print(arg)
    dataset = sys.argv[1]
    with open(f"{dataset_dir[dataset]}/reviews.pickle", "rb") as f:
        reviews = pickle.load(f)
    ret = get_aspect_count(dataset)
    df = pd.DataFrame(ret, columns=["index", "category", "count"])
    print(df)
    for fold in range(1, 6):
        print(f"Processing {dataset} fold {fold}")
        ret = get_aspect_count(dataset, fold)
        df = pd.DataFrame(ret, columns=["index", "category", "count"])
        output_path = f"{dataset_dir[dataset]}/{fold}/aspect_category_frequency.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
