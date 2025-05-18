from enum import Enum

bos = "<bos>"
eos = "<eos>"
pad = "<pad>"
fbos = "<feat>"


class StrEnum(Enum):
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


class SaveStrategy(StrEnum):
    BEST_TAG = "best_tag"
    BEST_TEXT = "best_text"
    ALL = "all"
