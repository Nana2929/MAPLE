"""
@File    :   run_latent.py
@Time    :   2024/06/09 01:18:55
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Latent embeddings Cosine-Sim & MSE Loss on MAPLE-generated explanations
"""

import numpy as np
import pandas as pd
import argparse


def cosine_similarity(A, B):
    assert A.shape == B.shape, "A and B must have the same shape"
    # Normalize the rows of A and B
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    # Compute pairwise cosine similarities
    cosine_similarities = np.sum(A_norm * B_norm, axis=1)
    return cosine_similarities


def mse_loss(A, B):
    return ((A - B) ** 2).mean(axis=-1)


def main(args):
    maple = {
        "query": np.load(f"{args.embeds_dir}/query.npy"),
    }
    # check shape
    print("Maple")
    for k, v in maple.items():
        print(k)
        print(v.shape)
    golden = np.load(f"{args.embeds_dir}/golden.npy")
    print("Golden", golden.shape)
    maple_mse_loss = mse_loss(maple["query"], golden)
    maple_cosine_sim = cosine_similarity(maple["query"], golden)
    # prag_mse_loss = mse_loss(prag["query"], golden)
    # prag_item_mse_loss = mse_loss(prag["item_query"], golden)
    # prag_z_mse_loss = mse_loss(prag["z"], golden)
    # prag_cosine_sim = cosine_similarity(prag["query"], golden)
    # prag_item_cosine_sim = cosine_similarity(prag["item_query"], golden)
    # prag_z_cosine_sim = cosine_similarity(prag["z"], golden)

    print("================ MSE Loss ================")
    print(f"Maple MSE Loss: {maple_mse_loss.mean()}")
    print("PRAG")
    # print(f"Query MSE Loss: {prag_mse_loss.mean()}")
    # print(f"Item-topic Query MSE Loss: {prag_item_mse_loss.mean()}")
    # print(f"Item-topic + Retrieved.mean() MSE Loss: {prag_z_mse_loss.mean()}")
    print("================ Cosine Similarity ================")
    print(f"Maple Cosine Similarity: {maple_cosine_sim.mean()}")
    print("PRAG")
    # print(f"Query Cosine Similarity: {prag_cosine_sim.mean()}")
    # print(f"Item-topic Query Cosine Similarity: {prag_item_cosine_sim.mean()}")
    # print(
    #     f"Item-topic + Retrieved.mean() Cosine Similarity: {prag_z_cosine_sim.mean()}"
    # )

    # df = [
    #     {
    #         "model": "Maple",
    #         "MSE Loss": maple_mse_loss.mean().item(),
    #         "Cosine Similarity": maple_cosine_sim.mean().item(),
    #     },
    #     {
    #         "model": "PRAG Query",
    #         "MSE Loss": prag_mse_loss.mean().item(),
    #         "Cosine Similarity": prag_cosine_sim.mean().item(),
    #     },
    #     {
    #         "model": "PRAG Item-topic Query",
    #         "MSE Loss": prag_item_mse_loss.mean().item(),
    #         "Cosine Similarity": prag_item_cosine_sim.mean().item(),
    #     },
    #     {
    #         "model": "PRAG Item-topic + Retrieved.mean()",
    #         "MSE Loss": prag_z_mse_loss.mean().item(),
    #         "Cosine Similarity": prag_z_cosine_sim.mean().item(),
    #     },
    # ]
    # df = pd.DataFrame(df)

    # df.to_csv(args.output, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeds_dir",
        type=str,
        default="./checkpoints/reproduce/yelp/1/embeds",
    )
    args = parser.parse_args()
    main(args)
