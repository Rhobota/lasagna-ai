import pytest

from enum import Enum
from lasagna.pydantic_util import ensure_pydantic_model

from pydantic import BaseModel

from typing import TypedDict
from typing_extensions import is_typeddict


class MyPydanticModel(BaseModel):
    a: str
    b: int


class MyColorEnum(Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'


class MyDict(TypedDict):
    name: str
    age: int
    color: MyColorEnum


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

    schema = m.model_json_schema()

    assert schema == {
        'type': 'object',
        'title': 'MyDictModel',
        'properties': {
            'name': {
                'default': None,
                'title': 'Name',
                'type': 'string',
            },
            'age': {
                'default': None,
                'title': 'Age',
                'type': 'integer',
            },
            'color': {
                '$ref': '#/$defs/MyColorEnum',
                'default': None,
            },
        },
        '$defs': {
            'MyColorEnum': {
                'enum': ['red', 'green', 'blue'],
                'title': 'MyColorEnum',
                'type': 'string',
            },
        },
    }
