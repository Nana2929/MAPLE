import os


class BaseDataInitializer:
    def __init__(
        self,
        data_path: os.PathLike,
        aspect_path: os.PathLike,
        index_dir: os.PathLike,
        *args,
        **kwargs,
    ):
        self.data_path = data_path
        self.aspect_path = aspect_path
        self.index_dir = index_dir
        self.data = None
        self.aspect_list = []
        self.aspect2idx = {}
        self.feature_set = set()
        self.max_rating = float("-inf")
        self.min_rating = float("inf")
        self.initialize_aspect()
        self.initialize()
        self.load_data()

    def initialize_aspect(self):
        """assign your self.aspect_list and self.aspect2idx"""
        raise NotImplementedError

    def initialize(self):
        """your user, item counting logic"""
        raise NotImplementedError

    def load_data(self, data_path, index_dir, *args, **kwargs):
        """load your data"""
        raise NotImplementedError

    def load_index(self, index_dir) -> tuple[list, list, list]:
        assert os.path.exists(index_dir), f"{index_dir} does not exist"
        with open(os.path.join(index_dir, "train.index"), "r") as f:
            train_index = [int(x) for x in f.readline().split(" ")]
        with open(os.path.join(index_dir, "validation.index"), "r") as f:
            valid_index = [int(x) for x in f.readline().split(" ")]
        with open(os.path.join(index_dir, "test.index"), "r") as f:
            test_index = [int(x) for x in f.readline().split(" ")]
        return train_index, valid_index, test_index
