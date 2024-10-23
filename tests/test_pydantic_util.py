import pytest

from enum import Enum
from lasagna.pydantic_util import ensure_pydantic_model

from pydantic import BaseModel, ValidationError

from typing import List
from typing_extensions import TypedDict, is_typeddict


class MyPydanticModel(BaseModel):
    a: str
    b: int


class MyColorEnum(Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'


class MySubDict(TypedDict):
    question: str
    answer: str


class MyDict(TypedDict):
    name: str
    age: int
    color: MyColorEnum
    about_me: List[MySubDict]


def test_model_is_already_pydantic():
    assert issubclass(MyPydanticModel, BaseModel)
    m = ensure_pydantic_model(MyPydanticModel)
    assert m is MyPydanticModel
    assert issubclass(m, BaseModel)


def test_model_unknown():
    with pytest.raises(ValueError):
        ensure_pydantic_model(int)


def test_convert_to_pydantic():
    assert is_typeddict(MyDict)
    m = ensure_pydantic_model(MyDict)
    assert not is_typeddict(m)
    assert issubclass(m, BaseModel)

    r = m(**{
        'name': 'ryan',
        'age': 84,
        'color': 'red',
        'about_me': [
            {
                'question': 'Why?',
                'answer': 'because',
            },
        ],
    })
    assert isinstance(r, m)
    assert r.name == 'ryan'  # type: ignore
    assert r.age == 84  # type: ignore
    assert r.color == MyColorEnum.RED  # type: ignore
    assert r.about_me[0]['question'] == 'Why?'  # type: ignore

    with pytest.raises(ValidationError):
        m(**{
            'name': 'ryan',
            'age': 'BAD VALUE!',
            'color': 'red',
            'about_me': [
                {
                    'question': 'Why?',
                    'answer': 'because',
                },
            ],
        })

    with pytest.raises(ValidationError):
        m(**{
            'name': 'ryan',
            'age': 84,
            'color': 'yellow',    # <-- yellow is not in the enum
            'about_me': [
                {
                    'question': 'Why?',
                    'answer': 'because',
                },
            ],
        })

    with pytest.raises(ValidationError):
        m(**{
            'name': 'ryan',
            'age': 84,
            'color': 'red',
            'about_me': [
                {
                    'question': 'Why?',
                    #'answer': 'because',   # <-- MISSING FIELD!
                },
            ],
        })

    with pytest.raises(ValidationError):
        m(**{
            'name': 'ryan',
            'age': 84,
            'color': 'red',
            'about_me': [
                {
                    'question': 'Why?',
                    'answer': {
                        'value': 'BAD SCHEMA!',
                    },
                },
            ],
        })

    schema = m.model_json_schema()

    assert schema == {
        "type": "object",
        "title": "MyDict",
        "properties": {
            "name": {
                "title": "Name",
                "type": "string"
            },
            "age": {
                "title": "Age",
                "type": "integer"
            },
            "color": {
                "$ref": "#/$defs/MyColorEnum"
            },
            "about_me": {
                "items": {
                    "$ref": "#/$defs/MySubDict"
                },
                "title": "About Me",
                "type": "array"
            }
        },
        "required": [
            "name",
            "age",
            "color",
            "about_me"
        ],
        "$defs": {
            "MyColorEnum": {
                "enum": [
                    "red",
                    "green",
                    "blue"
                ],
                "title": "MyColorEnum",
                "type": "string"
            },
            "MySubDict": {
                "type": "object",
                "title": "MySubDict",
                "properties": {
                    "question": {
                        "title": "Question",
                        "type": "string"
                    },
                    "answer": {
                        "title": "Answer",
                        "type": "string"
                    }
                },
                "required": [
                    "question",
                    "answer"
                ],
            }
        },
    }
