
import logging
import math
import string
import sys
from collections import defaultdict
from datetime import datetime

# check how to access the image
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from PIL import Image

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4')
# prepare a lemmatizer


YELP_IMG_DIR = "/home/P76114511/projects/yelp_2023_photos_tree"
GEST_IMG_DIR = "/home/P76114511/projects/Gest/images"


def batchify(data_size: int, batch_size: int = 100, shuffle: bool = False):

    # return indices
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, data_size, batch_size):
        yield indices[i : i + batch_size]


def get_yelp_image_path(image_name: str, root_dir=YELP_IMG_DIR) -> str:
    """Loading image from the image name following a dictionary structure

    Parameters
    ----------
    image_name : str, name of the image in the reviewdata (with file suffix)

    Returns
    -------
    img_final_path : str, path to the image
    """

    image_path = Path(root_dir)
    full_image_name = image_name
    for i in range(0, len(image_name), 2):
        image_path = image_path / image_name[i : i + 2]
        final_path = image_path / full_image_name
        if not full_image_name.endswith(".jpg"):
            final_path = final_path.with_suffix(".jpg")
        print(final_path)
        # check if the path exists
        if not final_path.exists():
            continue
        return str(final_path)


def get_gest_image_path(image_name: str, root_dir=GEST_IMG_DIR) -> str:
    """Loading image from the image name following a dictionary structure

    Parameters
    ----------
    image_name : str, name of the image in the reviewdata (without file suffix)

    Returns
    -------
    img_final_path : str, path to the image


    # images/AF1Qip/OV/0K/g4/AF1QipOV0Kg4GUKwQ2_rcAq0ASAXCJcF-tGACvqN7lI-.jpg
    # img_name = "AF1QipNb7nd3nww6uVOe9MaJzvrkiewlININEKYaXfRm"
    """

    image_path = Path(root_dir)
    prefix = "AF1Qip"
    full_image_name = image_name
    image_name = image_name[len(prefix) :]
    image_path = image_path / prefix
    for i in range(0, len(image_name), 2):
        image_path = image_path / image_name[i : i + 2]
        final_path = image_path / full_image_name
        final_path = final_path.with_suffix(".jpg")
        # print(final_path)
        # check if the path exists
        if not final_path.exists():
            continue
        return str(final_path)


def text_normalize(text: str, lemmatizer: WordNetLemmatizer) -> str:
    # if wnl is not defined, define it
    # if "wnl" not in globals():
    #     print("Initializing lemmatizer...")
    #     wnl = WordNetLemmatizer()
    text = text.lower()
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    text = lemmatizer.lemmatize(text)
    return text


def create_dual_logger(log_file_path: str, verbose: bool = False):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s %(module)s - %(message)s"
    )
    # Create a file handler and set the formatter
    file_handler = logging.FileHandler(log_file_path, mode="w+")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Create a stream handler to write to stdout and set the formatter
    if verbose:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

def merge_triplets(triplets:list[tuple],
                   senti2score:dict[str, float] = {'positive': 1., 'negative': -1., 'neutral': 0.},
                   ) -> tuple[str]:
        merged_text = set()
        merged_sentiments = []
        merged_rating = 0.
        merged_feat = set()
        # print(triplets)

        for triplet in triplets:
            feat, adj, text, sentiment= triplet

            merged_text.add(text)
            merged_sentiments.append(sentiment)
            merged_feat.add(feat)
            merged_rating += senti2score[sentiment]
        # check if any text is in another text
        new_merged_text = []
        for text in merged_text:
            for other_text in merged_text:
                if text != other_text and text in other_text:
                    break
            else:
                new_merged_text.append(text)
        merged_text = '. '.join(new_merged_text)
        merged_feat = list(merged_feat)
        return merged_text, merged_feat, merged_rating, merged_sentiments

class RsTrainTestSplitter:
    '''
    yelp'23 data format
    {'unique_id': 'bmfOFY1dW6YEm5UYNW3TdQ', 'user_id': 't2ZKf-CjGthLamYKNAcbJw', 'business_id': 'PLgwTcTuOPHvJhy7nGjx0A',
    'stars': 4.0, 'useful': 1,
     'funny': 0, 'cool': 2, 'text': 'I love this little neighborhood cafe and was reminded why on a recent lunch date there with a friend.
     The staff is really friendly and the space is bright and sunny, with healthy foods and tasty desserts. I was able to enjoy the small outdoor patio
     and my bean burger was pretty tasty. No real complaints, just a perfectly healthy lunch that is reasonably priced in an off-the-beaten-path
     storefront next to the quaint Demun business district.', 'date': '2019-08-11 20:29:19',
     'uie_absa': {'entity': {'offset': [['aspect', [21]], ['opinion', [24]], ['aspect', [27]],
     ['opinion', [29]], ['opinion', [31]], ['opinion', [34]], ['aspect', [35]], ['opinion', [37]], ['aspect', [38]],
    ...
     'event': {'offset': [], 'string': []},
     'triplet': [UIE_TRIPLET(sentiment='positive', aspect='staff', opinion='friendly'),
     UIE_TRIPLET(sentiment='positive', aspect='space', opinion='bright'), UIE_TRIPLET(sentiment='positive', aspect='space', opinion='sunny'), UIE_TRIPLET(sentiment='positive', aspect='food', opinion='healthy'), UIE_TRIPLET(sentiment='positive', aspect='dessert', opinion='tasty'), UIE_TRIPLET(sentiment='positive', aspect='bean burger', opinion='tasty'), UIE_TRIPLET(sentiment='positive', aspect='priced', opinion='reasonably')]}, 'tokens': ['I', 'love', 'this', 'little', 'neighborhood', 'cafe', 'and', 'was', 'reminded', 'why', 'on', 'a', 'recent', 'lunch', 'date', 'there', 'with', 'a', 'friend', '.', 'The', 'staff', 'is', 'really', 'friendly', 'and', 'the', 'space', 'is', 'bright', 'and', 'sunny', ',', 'with', 'healthy', 'foods', 'and', 'tasty', 'desserts', '.', 'I', 'was', 'able', 'to', 'enjoy', 'the', 'small', 'outdoor', 'patio', 'and', 'my', 'bean', 'burger', 'was', 'pretty', 'tasty', '.', 'No', 'real', 'complaints', ',', 'just', 'a', 'perfectly', 'healthy', 'lunch', 'that', 'is', 'reasonably', 'priced', 'in', 'an', 'off', '-', 'the', '-', 'beaten', '-', 'path', 'storefront', 'next', 'to', 'the', 'quaint', 'Demun', 'business', 'district', '.']}
    '''
    def __init__(self, data, test_ratio=0.1,
                 val_ratio=0.1,  random_state=42,
                 view='user',
                kfold=5):

        if kfold < 1:
            raise ValueError("kfold must be >= 1")

        self.data = data
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.view = view
        self.kfold = kfold

        self.user_reviews = defaultdict(list)
        self.item_reviews = defaultdict(list)
        for d in self.data:
            u = d.user
            i = d.item
            self.user_reviews[u].append(d.unique_id)
            self.item_reviews[i].append(d.unique_id)


    def __call__(self):
        return self.split(self.kfold)

    def split(self, kfold)->dict[str, list]:
        # 1. ensure that train set contains all users and items in the test/val set
        # 2. for each user/item's transaction history, split to train/val/test by random seed

        for kf in range(kfold):
            index_list = self._split(random_state=self.random_state + kf)
            logging.info(f"Splitting by kfold={kf}, random_state={self.random_state + kf},\
                         #train={len(index_list['train'])},\
                         #val={len(index_list['val'])}, #test={len(index_list['test'])}")
            yield index_list

    def _split(self, random_state=42)->dict[str, list]:
        index_list = {}
        index_list['train'] = set()
        # * 1. for each (user, item) pair in the trainset, the item needs to have at least 1 other entries
        # * so is the user
        def get_safe_size(total_size: int):
            if total_size <= 3: return total_size
            else: return int(total_size*0.5)

        for user, reviews in self.user_reviews.items():
            selected_rids = np.random.choice(reviews, size=get_safe_size(len(reviews)), replace=False)
            index_list['train'].update(selected_rids)
        for item, reviews in self.item_reviews.items():
            selected_rids = np.random.choice(reviews, size=get_safe_size(len(reviews)), replace=False)
            index_list['train'].update(selected_rids)

        # * 2. calculate the number of reviews in train/val/test set according to the ratio
        #*  rest_train, n_val, n_test
        n_train = math.ceil((1-self.test_ratio-self.val_ratio) * len(self.data))
        if n_train < len(index_list['train']):
            rest_train = 0
        else: rest_train = n_train - len(index_list['train'])
        # * 3. distribute rest of the reviews
        rest_reviews = [d for d in self.data if d.unique_id not in index_list['train']]
        print(f"rest u-i pairs after safeguarding warm trainset: {len(rest_reviews)}")
        # use val_ratio and test_ratio to split the rest of the reviews

        test_ratio = self.test_ratio / (self.test_ratio + self.val_ratio)
        val_ratio = 1 - test_ratio
        n_val = math.ceil(val_ratio * len(rest_reviews))

        print(f"n_val: {n_val}, n_test: {len(rest_reviews) - n_val}")
        # shuffle
        np.random.seed(random_state)
        np.random.shuffle(rest_reviews)
        # * 4. distribute rest of the reviews to train/val/test set
        index_list['train'] = list(index_list['train']) + [d.unique_id for d in rest_reviews[:rest_train]]
        index_list['val'] = [d.unique_id for d in rest_reviews[rest_train:rest_train+n_val]]
        index_list['test'] = [d.unique_id for d in rest_reviews[rest_train+n_val:]]
        return index_list


