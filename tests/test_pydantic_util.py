import pytest

from enum import Enum
from lasagna.pydantic_util import (
    ensure_pydantic_model,
    build_and_validate,
)

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


def test_ensure_pydantic_model_already_is_pydantic():
    assert issubclass(MyPydanticModel, BaseModel)
    m = ensure_pydantic_model(MyPydanticModel)
    assert m is MyPydanticModel
    assert issubclass(m, BaseModel)


def test_ensure_pydantic_model_unknown():
    with pytest.raises(ValueError):
        ensure_pydantic_model(int)


def test_ensure_pydantic_model_typeddict():
    assert is_typeddict(MyDict)
    m = ensure_pydantic_model(MyDict)
    assert not is_typeddict(m)
    assert issubclass(m, BaseModel)

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


def test_build_and_validate_already_is_pydantic():
    res = build_and_validate(
        MyPydanticModel,
        {
            'a': 'test',
            'b': 7,
        },
    )
    assert res.a == 'test'
    assert res.b == 7
    assert isinstance(res, MyPydanticModel)

    with pytest.raises(ValidationError):
        build_and_validate(
            MyPydanticModel,
            {},   # <-- empty payload!
        )

    with pytest.raises(ValidationError):
        build_and_validate(
            MyPydanticModel,
            {
                'a': 'test',
                #'b': 7,   # <-- missing field!
            },
        )

    with pytest.raises(ValidationError):
        build_and_validate(
            MyPydanticModel,
            {
                'a': 'test',
                'b': 'INVALID',   # <-- invalid value!
            },
        )


def test_build_and_validate_typeddict():
    res = build_and_validate(
        MyDict,
        {
            'name': 'ryan',
            'age': 84,
            'color': 'green',
            'about_me': [
                {
                    'question': 'Why?',
                    'answer': 'because',
                },
                {
                    'question': 'When?',
                    'answer': 'now',
                },
            ],
        },
    )
    assert res['name'] == 'ryan'
    assert res['age'] == 84
    assert res['color'] == MyColorEnum.GREEN
    assert res['about_me'][0]['question'] == 'Why?'
    assert res['about_me'][0]['answer'] == 'because'
    assert res['about_me'][1]['question'] == 'When?'
    assert res['about_me'][1]['answer'] == 'now'
    assert isinstance(res, dict)

    with pytest.raises(ValidationError):
        build_and_validate(
            MyDict,
            {},   # <-- empty payload!
        )

    with pytest.raises(ValidationError):
        build_and_validate(
            MyDict,
            {
                'name': 'ryan',
                'age': 84,
                'color': 'green',
                #'about_me': [],  # <-- missing field!
            },
        )

    with pytest.raises(ValidationError):
        build_and_validate(
            MyDict,
            {
                'name': 'ryan',
                'age': 84,
                'color': 'yellow',  # <-- bad emum!
                'about_me': [],
            },
        )

    with pytest.raises(ValidationError):
        build_and_validate(
            MyDict,
            {
                'name': 'ryan',
                'age': 'INVALID',   # <-- invalid value!
                'color': 'green',
                'about_me': [],
            },
        )

    with pytest.raises(ValidationError):
        build_and_validate(
            MyDict,
            {
                'name': 'ryan',
                'age': 84,
                'color': 'green',
                'about_me': [
                    {
                        'question': 'Why?',
                        #'answer': 'because',  # <-- missing nested field!
                    },
                ],
            },
        )

    with pytest.raises(ValidationError):
        build_and_validate(
            MyDict,
            {
                'name': 'ryan',
                'age': 84,
                'color': 'green',
                'about_me': [
                    {
                        'question': 'Why?',
                        'answer': {
                            'value': 'BAD SCHEMA',   # <-- invalid schema!
                        },
                    },
                ],
            },
        )
