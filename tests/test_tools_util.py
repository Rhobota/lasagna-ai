import pytest

from lasagna.types import (
    Message,
    ToolResult,
)

from typing import List, Dict, Callable

from lasagna.tools_util import (
    convert_to_json_schema,
    handle_tools,
    build_tool_response_message,
)


def test_convert_to_json_schema():
    j = convert_to_json_schema([
        {
            'name': 'a',
            'type': 'str',
            'description': 'param a',
        },
        {
            'name': 'second',
            'type': 'float',
            'description': 'second param',
        },
        {
            'name': 'another',
            'type': 'int',
            'description': 'another param',
        },
        {
            'name': 'yet_another',
            'type': 'bool',
            'description': 'yet another param',
        },
        {
            'name': 'an_enum',
            'type': 'enum cat dog house',
            'description': 'an enum param',
        },
        {
            'name': 'last_param',
            'type': 'bool',
            'description': '(optional) you can pass this if you want',
        },
    ])
    assert j == {
        'type': 'object',
        'properties': {
            'a': {
                'type': 'string',
                'description': 'param a',
            },
            'second': {
                'type': 'number',
                'description': 'second param',
            },
            'another': {
                'type': 'integer',
                'description': 'another param',
            },
            'yet_another': {
                'type': 'boolean',
                'description': 'yet another param',
            },
            'an_enum': {
                'type': 'string',
                'enum': ['cat', 'dog', 'house'],
                'description': 'an enum param',
            },
            'last_param': {
                'type': 'boolean',
                'description': '(optional) you can pass this if you want',
            },
        },
        'required': ['a', 'second', 'another', 'yet_another', 'an_enum'],
        'additionalProperties': False,
    }


def tool_a(first, second, third=5):
    return first + second + third

def tool_b(x):
    return x * 2

async def tool_async_a(x):
    return x * 3


@pytest.mark.asyncio
async def test_handle_tools():
    x = 4
    def tool_c():
        return x * 4

    async def tool_async_b():
        return x * 5

    tool_map: Dict[str, Callable] = {
        'tool_a': tool_a,
        'tool_b': tool_b,
        'tool_c': tool_c,
        'tool_async_a': tool_async_a,
        'tool_async_b': tool_async_b,
    }
    message: Message = {
        'role': 'tool_call',
        'tools': [
            {'call_id': '1001', 'function': {'arguments': '{"x": 8}', 'name': 'tool_b'}, 'call_type': 'function'},
            {'call_id': '1002', 'function': {'arguments': '{"x": 5.4}', 'name': 'tool_b'}, 'call_type': 'function'},
            {'call_id': '1003', 'function': {'arguments': '{"x": "hi"}', 'name': 'tool_b'}, 'call_type': 'function'},
            {'call_id': '1004', 'function': {'arguments': '{}', 'name': 'tool_b'}, 'call_type': 'function'},
            {'call_id': '1005', 'function': {'arguments': '{"y": "hi"}', 'name': 'tool_b'}, 'call_type': 'function'},
            {'call_id': '1006', 'function': {'arguments': '{"x": 4, "y": "hi"}', 'name': 'tool_b'}, 'call_type': 'function'},
            {'call_id': '1007', 'function': {'arguments': '{"first": 5, "second": 7.5}', 'name': 'tool_a'}, 'call_type': 'function'},
            {'call_id': '1008', 'function': {'arguments': '{"first": 5, "second": 7.5, "third": 3}', 'name': 'tool_a'}, 'call_type': 'function'},
            {'call_id': '1009', 'function': {'arguments': '{"third": 99, "first": 5, "second": 7.5}', 'name': 'tool_a'}, 'call_type': 'function'},
            {'call_id': '1010', 'function': {'arguments': '{}', 'name': 'tool_a'}, 'call_type': 'function'},
            {'call_id': '1011', 'function': {'arguments': '{"first": 5}', 'name': 'tool_a'}, 'call_type': 'function'},
            {'call_id': '1012', 'function': {'arguments': '{}', 'name': 'tool_c'}, 'call_type': 'function'},
            {'call_id': '1013', 'function': {'arguments': '{}', 'name': 'tool_d'}, 'call_type': 'function'},
            {'call_id': '1014', 'function': {'arguments': '{"x": -3}', 'name': 'tool_async_a'}, 'call_type': 'function'},
            {'call_id': '1015', 'function': {'arguments': '{}', 'name': 'tool_async_b'}, 'call_type': 'function'},
            {'call_id': '1016', 'function': {'arguments': '{}', 'name': 'tool_async_a'}, 'call_type': 'function'},
        ],
        'cost': None,
        'raw': None,
    }
    tool_results = await handle_tools([message], tool_map)
    assert tool_results is not None
    assert tool_results == [
        {'call_id': '1001', 'result': 16 },
        {'call_id': '1002', 'result': 10.8 },
        {'call_id': '1003', 'result': "hihi" },
        {'call_id': '1004', 'result': "TypeError: tool_b() missing 1 required positional argument: 'x'", 'is_error': True },
        {'call_id': '1005', 'result': "TypeError: tool_b() got an unexpected keyword argument 'y'", 'is_error': True },
        {'call_id': '1006', 'result': "TypeError: tool_b() got an unexpected keyword argument 'y'", 'is_error': True },
        {'call_id': '1007', 'result': 17.5 },
        {'call_id': '1008', 'result': 15.5 },
        {'call_id': '1009', 'result': 111.5 },
        {'call_id': '1010', 'result': "TypeError: tool_a() missing 2 required positional arguments: 'first' and 'second'", 'is_error': True },
        {'call_id': '1011', 'result': "TypeError: tool_a() missing 1 required positional argument: 'second'", 'is_error': True },
        {'call_id': '1012', 'result': 16 },
        {'call_id': '1013', 'result': "KeyError: 'tool_d'", 'is_error': True },
        {'call_id': '1014', 'result': -9 },
        {'call_id': '1015', 'result': 20 },
        {'call_id': '1016', 'result': "TypeError: tool_async_a() missing 1 required positional argument: 'x'", 'is_error': True },
    ]


def test_build_tool_response_message():
    res: List[ToolResult] = [
        {'call_id': '1002', 'result': 10.8 },
        {'call_id': '1003', 'result': "hihi" },
    ]
    message = build_tool_response_message(res)
    assert message == {
        'role': 'tool_res',
        'tools': res,
    }
