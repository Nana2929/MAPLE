from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

import random
from .utils import identity_tokenizer, run_tfidf, get_tfidf_score


def simple_load_test_data(data: list[any], test_index: list[int]):
    # for hit_ratio and ndcg calculation
    test = []
    for idx in test_index:
        d = data[idx]
        test.append(d)
    return test


def random_load_test_data(
    data: list[any], test_index: list[int], aspect_list: list[str], topk: int, seed=42
):
    random.seed(seed)
    test = []

    C = len(aspect_list)

    for idx in test_index:
        d = data[idx]
        fc = random.sample(range(C), topk)
        fcn = [aspect_list[i] for i in fc]
        assert len(fc) > 0 and len(fc) == len(fcn)
        d["fake_categories"] = fc
        d["fake_category_names"] = fcn
        test.append(d)
    return test


def heuristic_load_test_data(
    data: list[any],
    test_index: list[int],
    user2cat: dict[int, list[int]],
    item2cat: dict[int, list[int]],
    aspect2idx: dict[str, int],
    aspect_list: list[str],
    topk: int,
):
    test = []
    for idx in test_index:
        u, i = data[idx]["user"], data[idx]["item"]
        intersect = lambda a, b: list(set(a) & set(b))
        fc = intersect(user2cat.get(u, []), item2cat.get(i, []))

        if len(fc) == 0 and len(item2cat.get(i, [])) > 0:
            fc = [random.choice(list(item2cat[i]))]
        if len(item2cat.get(i, [])) == 0:
            fc = [random.randint(0, len(aspect_list) - 1)]

        # trimming the fake tokens
        fc = fc[:topk]
        fcn = [aspect_list[i] for i in fc]
        data[idx]["fake_categories"] = fc
        data[idx]["fake_category_names"] = fcn
        test.append(data[idx])
    return test


def tfidf_load_test_data(
    data: list[any],
    test_index: list[int],
    nuser: int,
    nitem: int,
    aspect_list: list[str],
    topk: int,
    user2cat: dict[int, set[int]],
    item2cat: dict[int, set[int]],
):
    test = []

    tfidf = TfidfVectorizer(
        tokenizer=identity_tokenizer,
        lowercase=False,
        min_df=0.001,
    )
    i_cat = [None] * nitem
    u_cat = [None] * nuser
    cold_start_items = 0
    cold_start_users = 0
    for user_index in range(nuser):
        res = user2cat.get(user_index, None)
        if res is None:
            cold_start_users += 1
        u_cat[user_index] = res if res is not None else []
    for item_index in range(nitem):
        res = item2cat.get(item_index, None)
        if res is None:
            cold_start_items += 1
        i_cat[item_index] = res if res is not None else []

    user_tfidf = run_tfidf(u_cat, vectorizer=tfidf)
    item_tfidf = run_tfidf(i_cat, vectorizer=tfidf)

    fc_count = 0
    for idx in test_index:
        u, i = data[idx]["user"], data[idx]["item"]
        u_tfidf = get_tfidf_score(
            id=u,
            tfidf=user_tfidf,
            topk=None,
            return_score=True,
        )
        i_tfidf = get_tfidf_score(
            id=i,
            tfidf=item_tfidf,
            topk=None,
            return_score=True,
        )
        score_dict = defaultdict(float)
        for f, s in u_tfidf:
            score_dict[f] += s
        for f, s in i_tfidf:
            score_dict[f] += s
        # sort and get the topk (descending order)
        sorted_score_dict = sorted(score_dict.items(), key=lambda x: -x[1])
        fc = [f for f, _ in sorted_score_dict]
        if len(fc) == 0:
            fc = [random.randint(0, len(aspect_list) - 1)]

        # slicing to max_test_aspect_tokens
        fc = fc[:topk]
        fc_count += len(fc)

        fcn = [aspect_list[i] for i in fc]
        data[idx]["fake_categories"] = fc
        data[idx]["fake_category_names"] = fcn
        test.append(data[idx])
    print("cold_start_users: ", cold_start_users)
    print("cold_start_items: ", cold_start_items)
    print("averaged tfidf fake-category count: ", fc_count / len(test_index))
    return test
