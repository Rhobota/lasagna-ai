import pytest

from lasagna.util import (
    parse_docstring,
    combine_pairs,
)

from typing import List


def test_parse_docstring():
    desc, params = parse_docstring("""
        This is the description
        that continues here.
          with indented stuff here
        back to normal.

        :param: x: str: the `x` value

        :param: another: int: another value
    """)
    assert desc == "This is the description that continues here.   with indented stuff here back to normal."
    assert params == [
        {
            'name': 'x',
            'type': 'str',
            'description': 'the `x` value',
        },
        {
            'name': 'another',
            'type': 'int',
            'description': 'another value',
        },
    ]

    desc, params = parse_docstring("""Hi
        this is the docs.
        :param: x: str: the `x` value
        :param: another: int: another value
    """)
    assert desc == "Hi this is the docs."
    assert params == [
        {
            'name': 'x',
            'type': 'str',
            'description': 'the `x` value',
        },
        {
            'name': 'another',
            'type': 'int',
            'description': 'another value',
        },
    ]

    desc, params = parse_docstring("""
        Hi this is the docs.
        :param: x: str: the `x` value
        :param: another: int: another value
    """)
    assert desc == "Hi this is the docs."
    assert params == [
        {
            'name': 'x',
            'type': 'str',
            'description': 'the `x` value',
        },
        {
            'name': 'another',
            'type': 'int',
            'description': 'another value',
        },
    ]

    desc, params = parse_docstring("""
        Hi this is the docs.

        :param: x: str: the `x` value
        :param: another: int: another value
    """)
    assert desc == "Hi this is the docs."
    assert params == [
        {
            'name': 'x',
            'type': 'str',
            'description': 'the `x` value',
        },
        {
            'name': 'another',
            'type': 'int',
            'description': 'another value',
        },
    ]

    desc, params = parse_docstring("""
        Hi this is the
        docs.


        :param: x: str: the `x` value
        :param: another: int: another value
    """)
    assert desc == "Hi this is the docs."
    assert params == [
        {
            'name': 'x',
            'type': 'str',
            'description': 'the `x` value',
        },
        {
            'name': 'another',
            'type': 'int',
            'description': 'another value',
        },
    ]

    desc, params = parse_docstring("""
        Hi this is the
        docs.


        :param: x: str: the `x` value

        :param: another: int: another value
    """)
    assert desc == "Hi this is the docs."
    assert params == [
        {
            'name': 'x',
            'type': 'str',
            'description': 'the `x` value',
        },
        {
            'name': 'another',
            'type': 'int',
            'description': 'another value',
        },
    ]

    desc, params = parse_docstring("""Hi this is the docs.
        :param: x: str: the `x` value
        :param: another: int: another value
    """)
    assert desc == "Hi this is the docs."
    assert params == [
        {
            'name': 'x',
            'type': 'str',
            'description': 'the `x` value',
        },
        {
            'name': 'another',
            'type': 'int',
            'description': 'another value',
        },
    ]

    desc, params = parse_docstring("""Hi this is the docs.

        :param: x: str: the `x` value
        :param: another: int: another value
    """)
    assert desc == "Hi this is the docs."
    assert params == [
        {
            'name': 'x',
            'type': 'str',
            'description': 'the `x` value',
        },
        {
            'name': 'another',
            'type': 'int',
            'description': 'another value',
        },
    ]

    desc, params = parse_docstring("""
        Hi this is the docs.
        :param: thing: float: (optional) the thing
        :param: another: enum a b other: another value
    """)
    assert desc == "Hi this is the docs."
    assert params == [
        {
            'name': 'thing',
            'type': 'float',
            'description': '(optional) the thing',
        },
        {
            'name': 'another',
            'type': 'enum a b other',
            'description': 'another value',
        },
    ]

    with pytest.raises(ValueError):
        desc, params = parse_docstring("")

    with pytest.raises(ValueError):
        desc, params = parse_docstring("""
            :param: thing: float: (optional) the thing
            :param: another: enum a b other: another value
        """)

    with pytest.raises(ValueError):
        desc, params = parse_docstring("""

            :param: thing: float: (optional) the thing
            :param: another: enum a b other: another value
        """)

    with pytest.raises(ValueError):
        desc, params = parse_docstring("""
            Docs
            :param: x: badtype: the param
        """)
        print(desc)
        print(params)


def test_combine_pairs():
    def combine_rule_1(a, b):
        return False
    def combine_rule_2(a, b):
        return True, 99
    def combine_rule_3(a, b):
        if a == b:
            return True, a
        return False

    empty: List[int] = []
    oneval: List[int] = [7]
    twovals: List[int] = [7, 12]
    threevals: List[int] = [7, 12, 4]

    assert combine_pairs(empty, combine_rule_1) == []
    assert combine_pairs(empty, combine_rule_2) == []
    assert combine_pairs(empty, combine_rule_3) == []

    assert combine_pairs(oneval, combine_rule_1) == [7]
    assert combine_pairs(oneval, combine_rule_2) == [7]
    assert combine_pairs(oneval, combine_rule_3) == [7]

    assert combine_pairs(twovals, combine_rule_1) == [7, 12]
    assert combine_pairs(twovals, combine_rule_2) == [99]
    assert combine_pairs(twovals, combine_rule_3) == [7, 12]

    assert combine_pairs(threevals, combine_rule_1) == [7, 12, 4]
    assert combine_pairs(threevals, combine_rule_2) == [99, 99]
    assert combine_pairs(threevals, combine_rule_3) == [7, 12, 4]

    assert combine_pairs([1, 4], combine_rule_3) == [1, 4]
    assert combine_pairs([4, 4], combine_rule_3) == [4]

    assert combine_pairs([1, 4, 7], combine_rule_3) == [1, 4, 7]
    assert combine_pairs([4, 4, 7], combine_rule_3) == [4, 7]
    assert combine_pairs([1, 4, 4], combine_rule_3) == [1, 4]
    assert combine_pairs([4, 4, 4], combine_rule_3) == [4, 4]

    assert combine_pairs([1, 4, 7, 9], combine_rule_3) == [1, 4, 7, 9]
    assert combine_pairs([4, 4, 7, 9], combine_rule_3) == [4, 7, 9]
    assert combine_pairs([1, 4, 4, 9], combine_rule_3) == [1, 4, 9]
    assert combine_pairs([1, 4, 7, 7], combine_rule_3) == [1, 4, 7]
    assert combine_pairs([4, 4, 7, 7], combine_rule_3) == [4, 7]
    assert combine_pairs([4, 4, 4, 9], combine_rule_3) == [4, 4, 9]
    assert combine_pairs([1, 7, 7, 7], combine_rule_3) == [1, 7, 7]
    assert combine_pairs([7, 7, 7, 7], combine_rule_3) == [7, 7, 7]
