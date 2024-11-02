import pytest

from lasagna.util import (
    parse_docstring,
    combine_pairs,
    convert_to_image_url,
    convert_to_image_base64,
    exponential_backoff_retry_delays,
    get_name,
    recursive_hash,
)

from typing import List

import tempfile
import hashlib
import os


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
            'optional': True,
        },
        {
            'name': 'another',
            'type': 'enum a b other',
            'description': 'another value',
        },
    ]

    desc, params = parse_docstring("""
        Hi this is the docs.
        And this is a second line.
    """)
    assert desc == "Hi this is the docs. And this is a second line."
    assert params == []

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


@pytest.mark.asyncio
async def test_convert_to_image_url():
    url = 'https://example.com/img.png'
    s = await convert_to_image_url(url)
    assert s == url

    url = 'http://example.com/img.png'
    s = await convert_to_image_url(url)
    assert s == url

    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'a.jpeg')
        with open(fn, 'wb') as f:
            f.write(b'1234')
        s = await convert_to_image_url(fn)
        assert s == 'data:image/jpeg;base64,MTIzNA=='

        fn = os.path.join(tmp, 'a.png')
        with open(fn, 'wb') as f:
            f.write(b'1235')
        s = await convert_to_image_url(fn)
        assert s == 'data:image/png;base64,MTIzNQ=='

        fn = os.path.abspath(fn)
        with open(fn, 'wb') as f:
            f.write(b'1235')
        s = await convert_to_image_url(fn)
        assert s == 'data:image/png;base64,MTIzNQ=='

        fn_with_schema = f"file://{fn}"
        with open(fn, 'wb') as f:
            f.write(b'1235')
        s = await convert_to_image_url(fn_with_schema)
        assert s == 'data:image/png;base64,MTIzNQ=='

    with pytest.raises(ValueError):
        await convert_to_image_url(os.path.join('does', 'not', 'exist.png'))


@pytest.mark.asyncio
async def test_convert_to_image_base64():
    # Disabled so that our tests don't make remote calls.
    #url = 'https://raw.githubusercontent.com/Rhobota/lasagna-ai/main/logos/lasagna-ai.png'
    #mimetype, s = await convert_to_image_base64(url)
    #assert mimetype == 'image/png'
    #assert len(s) > 1000

    # Disabled so that our tests don't make remote calls.
    #with pytest.raises(Exception):
    #    url = 'https://raw.githubusercontent.com/Rhobota/lasagna-ai/main/logos/DOES_NOT_EXIST.png'
    #    await convert_to_image_base64(url)

    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'a.jpeg')
        with open(fn, 'wb') as f:
            f.write(b'1234')
        mimetype, s = await convert_to_image_base64(fn)
        assert mimetype == 'image/jpeg'
        assert s == 'MTIzNA=='

        fn = os.path.join(tmp, 'a.png')
        with open(fn, 'wb') as f:
            f.write(b'1235')
        mimetype, s = await convert_to_image_base64(fn)
        assert mimetype == 'image/png'
        assert s == 'MTIzNQ=='

        fn = os.path.abspath(fn)
        with open(fn, 'wb') as f:
            f.write(b'1235')
        mimetype, s = await convert_to_image_base64(fn)
        assert mimetype == 'image/png'
        assert s == 'MTIzNQ=='

        fn_with_schema = f"file://{fn}"
        with open(fn, 'wb') as f:
            f.write(b'1235')
        mimetype, s = await convert_to_image_base64(fn_with_schema)
        assert mimetype == 'image/png'
        assert s == 'MTIzNQ=='

    with pytest.raises(ValueError):
        await convert_to_image_base64(os.path.join('does', 'not', 'exist.png'))


def test_exponential_backoff_retry_delays():
    with pytest.raises(AssertionError):
        exponential_backoff_retry_delays(0)
    assert exponential_backoff_retry_delays(1, 3.0, 1e10) == [0.0]
    assert exponential_backoff_retry_delays(2, 3.0, 1e10) == [3.0, 0.0]
    assert exponential_backoff_retry_delays(3, 3.0, 1e10) == [3.0, 9.0, 0.0]
    assert exponential_backoff_retry_delays(4, 3.0, 1e10) == [3.0, 9.0, 27.0, 0.0]
    assert exponential_backoff_retry_delays(4, 5.0, 1e10) == [5.0, 25.0, 125.0, 0.0]
    assert exponential_backoff_retry_delays(4, 5.0, 30.0) == [5.0, 25.0, 30.0, 0.0]


def regular_function():
    pass

def async_regular_function():
    pass

class class_with_str_method:
    def __str__(self) -> str:
        return 'Hi!'

def test_get_name():
    assert get_name(regular_function) == 'regular_function'
    assert get_name(async_regular_function) == 'async_regular_function'
    assert get_name(class_with_str_method()) == 'Hi!'


def test_recursive_hash():
    obj = {
        'b': -5,
        'a': [4, True, False, 'hi', (3.5, 7.0), None],
    }
    got = recursive_hash('hola', obj)
    s = ''.join([v.strip() for v in '''
        __str__hola
        __open_dict__
            __open_list__
                __open_list__
                    __str__a
                    __open_list__
                        __int__4
                        __bool__1
                        __bool__0
                        __str__hi
                        __open_list__
                            __float__3.50000000e+00
                            __float__7.00000000e+00
                        __close_list__
                        __None__
                    __close_list__
                __close_list__
                __open_list__
                    __str__b
                    __int__-5
                __close_list__
            __close_list__
        __close_dict__
        '''.strip().splitlines()])
    correct = hashlib.sha256(s.encode('utf-8')).hexdigest()
    assert got == correct
