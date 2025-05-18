import datetime
import math
import re

###############################################################################
# Eval funcs needed during training, data-postprocessing functions
###############################################################################


def evaluate_hit_ratio(user2items_test, user2items_top, top_k=None):
    hits = 0
    for label, predict in zip(user2items_test, user2items_top):

        if top_k is not None:
            try:
                predict = predict[
                    :top_k
                ]  # assuming the predict is ordered by rank score
            except:
                print("label", label)
                print("predict", predict)

        rank_list = set(predict)
        test_list = set(label)
        if len(rank_list & test_list) > 0:
            hits += 1
    return hits / len(user2items_test)


def evaluate_ndcg(user2items_test, user2items_top, top_k=20):
    # https://github.com/REASONER2023/reasoner2023.github.io/blob/main/metrics/metrics.py#L62
    dcgs = [1 / math.log(i + 2) for i in range(top_k)]

    ndcg = 0
    for i in range(len(user2items_test)):
        rank_list = user2items_top[i]
        test_list = user2items_test[i]
        dcg_u = 0
        for idx, item in enumerate(rank_list):
            if idx >= top_k:
                break
            if item in test_list:
                dcg_u += dcgs[idx]

        ndcg += dcg_u

    return ndcg / (sum(dcgs) * len(user2items_test))


def evaluate_precision_recall_f1(
    top_k, user2items_test, user2items_top
):  # (B, pos_num)  (B, k)
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    for i in range(len(user2items_test)):
        rank_list = user2items_top[i]
        test_list = user2items_test[i]
        hits = len(set(test_list) & set(rank_list))
        pre = hits / top_k
        rec = hits / len(test_list)
        precision_sum += pre
        recall_sum += rec
        if (pre + rec) > 0:
            f1_sum += 2 * pre * rec / (pre + rec)

    precision = precision_sum / len(user2items_test)
    recall = recall_sum / len(user2items_test)
    f1 = f1_sum / len(user2items_test)

    return precision, recall, f1


def getstem(word, lemmatizer=None):
    return lemmatizer.lemmatize(word)


def feature_detect(seq_batch, feature_set):
    lemmatizer = WordNetLemmatizer()
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            i = getstem(i, lemmatizer)
            if i in feature_set:
                feature_list.append(i)

        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for fea_set, fea in zip(feature_batch, test_feature):
        if isinstance(fea, list):
            for f in fea:  # a list of features
                if f in fea_set:
                    count += 1

        else:  # single feature
            if fea in fea_set:
                count += 1
    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for r, p in predicted:
        p = max(p, max_r)
        p = min(p, min_r)
        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub**2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


def now_time():
    return "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "]: "


def postprocessing(string):
    """
    adopted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub("'s", " 's", string)
    string = re.sub("'m", " 'm", string)
    string = re.sub("'ve", " 've", string)
    string = re.sub("n't", " n't", string)
    string = re.sub("'re", " 're", string)
    string = re.sub("'d", " 'd", string)
    string = re.sub("'ll", " 'll", string)
    string = re.sub("\(", " ( ", string)
    string = re.sub("\)", " ) ", string)
    string = re.sub(",+", " , ", string)
    string = re.sub(":+", " , ", string)
    string = re.sub(";+", " . ", string)
    string = re.sub("\.+", " . ", string)
    string = re.sub("!+", " ! ", string)
    string = re.sub("\?+", " ? ", string)
    string = re.sub(" +", " ", string).strip()
    return string


def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids)
    text = postprocessing(text)  # process punctuations: "good!" -> "good !"
    tokens = []
    for token in text.split():
        if token == eos:
            break
        tokens.append(token)
    return tokens
