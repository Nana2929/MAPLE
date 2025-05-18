from enum import Enum, EnumMeta


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


class Strategy(BaseEnum):
    # https://stackoverflow.com/questions/63335753/how-to-check-if-string-exists-in-enum-of-strings
    HEURISTIC = "heuristic"
    SUPERVISED = "supervised"
    GT = "gt"
