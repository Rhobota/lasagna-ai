import pytest

from lasagna.agent_util import (
    bind_model,
    build_most_simple_agent,
)

from lasagna.mock_provider import (
    MockProvider,
)

from lasagna.types import (
    AgentCallable,
    BoundAgentCallable,
    Message,
    ToolResult,
)

from typing import List, Dict, Callable, Awaitable

from lasagna.tools_util import (
    convert_to_json_schema,
    get_name,
    handle_tools,
    build_tool_response_message,
    is_async_callable,
    is_callable_of_type,
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
            'optional': True,
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
        {'type': 'any', 'call_id': '1001', 'result': 16 },
        {'type': 'any', 'call_id': '1002', 'result': 10.8 },
        {'type': 'any', 'call_id': '1003', 'result': "hihi" },
        {'type': 'any', 'call_id': '1004', 'result': "TypeError: tool_b() missing 1 required positional argument: 'x'", 'is_error': True },
        {'type': 'any', 'call_id': '1005', 'result': "TypeError: tool_b() got an unexpected keyword argument 'y'", 'is_error': True },
        {'type': 'any', 'call_id': '1006', 'result': "TypeError: tool_b() got an unexpected keyword argument 'y'", 'is_error': True },
        {'type': 'any', 'call_id': '1007', 'result': 17.5 },
        {'type': 'any', 'call_id': '1008', 'result': 15.5 },
        {'type': 'any', 'call_id': '1009', 'result': 111.5 },
        {'type': 'any', 'call_id': '1010', 'result': "TypeError: tool_a() missing 2 required positional arguments: 'first' and 'second'", 'is_error': True },
        {'type': 'any', 'call_id': '1011', 'result': "TypeError: tool_a() missing 1 required positional argument: 'second'", 'is_error': True },
        {'type': 'any', 'call_id': '1012', 'result': 16 },
        {'type': 'any', 'call_id': '1013', 'result': "KeyError: 'tool_d'", 'is_error': True },
        {'type': 'any', 'call_id': '1014', 'result': -9 },
        {'type': 'any', 'call_id': '1015', 'result': 20 },
        {'type': 'any', 'call_id': '1016', 'result': "TypeError: tool_async_a() missing 1 required positional argument: 'x'", 'is_error': True },
    ]


def test_build_tool_response_message():
    res: List[ToolResult] = [
        {'type': 'any', 'call_id': '1002', 'result': 10.8 },
        {'type': 'any', 'call_id': '1003', 'result': "hihi" },
    ]
    message = build_tool_response_message(res)
    assert message == {
        'role': 'tool_res',
        'tools': res,
    }


MyCallable = Callable[[int, str], Awaitable[bool]]

async def correct_function(a: int, b: str) -> bool:
    return True

async def bad_a_type(a: str, b: str) -> bool:
    return True

async def missing_parameter(a: int) -> bool:
    return True

async def extra_parameter(a: int, b: str, c: int) -> bool:
    return True

async def bad_return_type(a: int, b: str) -> int:
    return True

async def missing_return_type(a: int, b: str):
    return True

async def missing_a_type(a, b: str) -> bool:
    return True

def not_async(a: int, b: str) -> bool:
    return True

class correct_class:
    async def __call__(self, a: int, b: str) -> bool:
        return True

class bad_a_type_class:
    async def __call__(self, a: str, b: str) -> bool:
        return True

class missing_parameter_class:
    async def __call__(self, a: int) -> bool:
        return True

class extra_parameter_class:
    async def __call__(self, a: int, b: str, c: int) -> bool:
        return True

class bad_return_type_class:
    async def __call__(self, a: int, b: str) -> int:
        return True

class missing_return_type_class:
    async def __call__(self, a: int, b: str):
        return True

class missing_a_type_class:
    async def __call__(self, a, b: str) -> bool:
        return True

class not_async_class:
    def __call__(self, a: int, b: str) -> bool:
        return True

class class_with_str_method:
    def __str__(self) -> str:
        return 'Hi!'


def test_get_name():
    assert get_name(correct_function) == 'correct_function'
    assert get_name(class_with_str_method()) == 'Hi!'


def test_is_async_callable():
    assert is_async_callable(correct_function)
    assert not is_async_callable(not_async)

    assert is_async_callable(correct_class())
    assert not is_async_callable(not_async_class())


def test_is_callable_of_type():
    objects = [
        correct_function,
        bad_a_type,
        missing_parameter,
        extra_parameter,
        bad_return_type,
        missing_return_type,
        missing_a_type,
        not_async,
        42,
        "not a function",
        correct_class(),
        bad_a_type_class(),
        missing_parameter_class(),
        extra_parameter_class(),
        bad_return_type_class(),
        missing_return_type_class(),
        missing_a_type_class(),
        not_async_class(),
    ]

    matching_objects = [
        o
        for o in objects
        if is_callable_of_type(o, MyCallable, no_throw=True)
    ]

    assert len(matching_objects) == 2
    assert matching_objects[0] is correct_function
    assert isinstance(matching_objects[1], correct_class)


def test_is_callable_of_type_agent_callables():
    my_binder = bind_model(MockProvider, 'some_model', {'a': 'yes', 'b': 6})
    my_agent = build_most_simple_agent()
    my_bound_agent = my_binder(my_agent)

    assert is_callable_of_type(my_agent, AgentCallable)
    assert is_callable_of_type(my_bound_agent, BoundAgentCallable)

    assert not is_callable_of_type(my_agent, BoundAgentCallable)
    assert not is_callable_of_type(my_bound_agent, AgentCallable)
