import pytest

from lasagna.util import parse_docstring


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


# TODO: test combine_pairs
