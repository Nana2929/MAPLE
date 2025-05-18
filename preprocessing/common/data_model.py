from collections import namedtuple
from dataclasses import asdict, dataclass

UIE_TRIPLET = namedtuple("UIE_TRIPLET", ["aspect", "opinion", "text", "sentiment"])


@dataclass
class yelp_review:
    review_id: str
    user_id: str
    business_id: str
    stars: int
    useful: int
    funny: int
    cool: int
    text: str
    date: str
    uie_absa: any
    tokens: list[str]

    def to_dict(self):
        return asdict(self)


@dataclass
class yelp_tip:
    text: str
    date: str
    compliment_count: int  # likes
    business_id: str
    user_id: str

    def to_dict(self):
        return asdict(self)


@dataclass
class yelp_business:
    business_id: str
    name: str
    address: str
    city: str
    state: str
    postal_code: str
    latitude: float
    longitude: float
    stars: float
    review_count: int
    is_open: int
    attributes: dict
    categories: list[str]
    hours: dict

    def to_dict(self):
        return asdict(self)


# dict_keys(['user_id', 'name', 'review_count', 'yelping_since', 'useful', 'funny', 'cool', 'elite', 'friends', 'fans', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos'])
@dataclass
class yelp_user:
    user_id: str
    name: str
    review_count: int
    yelping_since: str
    useful: int
    funny: int
    cool: int
    elite: str
    friends: list[str]
    fans: int
    average_stars: float
    compliment_hot: int
    compliment_more: int
    compliment_profile: int
    compliment_cute: int
    compliment_list: int
    compliment_note: int
    compliment_plain: int
    compliment_cool: int
    compliment_funny: int
    compliment_writer: int
    compliment_photos: int

    def to_dict(self):
        return asdict(self)
