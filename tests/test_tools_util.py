import pytest

from lasagna.agent_util import (
    bind_model,
    build_simple_agent,
    noop_callback,
)

from lasagna.mock_provider import (
    MockProvider,
)

from lasagna.types import (
    AgentCallable,
    BoundAgentCallable,
    Message,
    ModelSpec,
    ToolResult,
)

from typing import List, Dict, Callable, Awaitable

from lasagna.tools_util import (
    convert_to_json_schema,
    get_tool_params,
    handle_tools,
    build_tool_response_message,
    is_async_callable,
    is_callable_of_type,
    extract_tool_result_as_sting,
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
async def test_handle_tools_standard_functions():
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
    tool_results = await handle_tools(
        prev_messages = [],
        new_messages = [message],
        tools_map = tool_map,
        event_callback = noop_callback,
        model_spec = None,
    )
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


@pytest.mark.asyncio
async def test_handle_tools_layered_agents():
    outer_model_spec: ModelSpec = {
        'provider': MockProvider,
        'model': 'some_model',
        'model_kwargs': {'outer': True},
    }

    inner_model_spec: ModelSpec = {
        'provider': MockProvider,
        'model': 'some_model',
        'model_kwargs': {'outer': False},
    }

    agent_1 = build_simple_agent(name = 'agent_1')
    my_binder = bind_model(**inner_model_spec)
    bound_agent_2 = my_binder(build_simple_agent(name = 'bound_agent_2'))

    tool_map: Dict[str, Callable] = {
        'agent_1': agent_1,
        'bound_agent_2': bound_agent_2,
    }

    message: Message = {
        'role': 'tool_call',
        'tools': [
            {'call_id': '1001', 'function': {'arguments': '{}', 'name': 'agent_1'}, 'call_type': 'function'},
            {'call_id': '1002', 'function': {'arguments': '{}', 'name': 'bound_agent_2'}, 'call_type': 'function'},
            {'call_id': '1003', 'function': {'arguments': '{"prompt": "Hi"}', 'name': 'agent_1'}, 'call_type': 'function'},
            {'call_id': '1004', 'function': {'arguments': '{"prompt": "Hey"}', 'name': 'bound_agent_2'}, 'call_type': 'function'},
        ],
    }

    prev_messages: List[Message] = [
        {
            'role': 'system',
            'text': 'You are AI.',
        },
    ]

    tool_results = await handle_tools(
        prev_messages = prev_messages,
        new_messages = [message],
        tools_map = tool_map,
        event_callback = noop_callback,
        model_spec = outer_model_spec,
    )

    assert prev_messages == [  # test immutability of prev_messages
        {
            'role': 'system',
            'text': 'You are AI.',
        },
    ]

    assert tool_results is not None
    assert tool_results == [
        {
            'type': 'layered_agent',
            'call_id': '1001',
            'run': {
                'type': 'messages',
                'agent': 'agent_1',
                'provider': 'MockProvider',
                'model': 'some_model',
                'model_kwargs': {'outer': True},
                'messages': [
                    {
                        'role': 'system',
                        'text': 'You are AI.',
                    },
                    {
                        'role': 'ai',
                        'text': 'model: some_model',
                    },
                    {
                        'role': 'human',
                        'text': 'model_kwarg: outer = True',
                    },
                ],
            },
        },
        {
            'type': 'layered_agent',
            'call_id': '1002',
            'run': {
                'type': 'messages',
                'agent': 'bound_agent_2',
                'provider': 'MockProvider',
                'model': 'some_model',
                'model_kwargs': {'outer': False},
                'messages': [
                    {
                        'role': 'system',
                        'text': 'You are AI.',
                    },
                    {
                        'role': 'ai',
                        'text': 'model: some_model',
                    },
                    {
                        'role': 'human',
                        'text': 'model_kwarg: outer = False',
                    },
                ],
            },
        },
        {
            'type': 'layered_agent',
            'call_id': '1003',
            'run': {
                'type': 'messages',
                'agent': 'agent_1',
                'provider': 'MockProvider',
                'model': 'some_model',
                'model_kwargs': {'outer': True},
                'messages': [
                    {
                        'role': 'system',
                        'text': 'You are AI.',
                    },
                    {
                        'role': 'tool_call',
                        'tools': [
                            {
                                'call_id': '1003',
                                'call_type': 'function',
                                'function': {
                                    'arguments': '{"prompt": "Hi"}',
                                    'name': 'agent_1',
                                },
                            },
                        ],
                    },
                    {
                        'role': 'ai',
                        'text': 'model: some_model',
                    },
                    {
                        'role': 'human',
                        'text': 'model_kwarg: outer = True',
                    },
                ],
            },
        },
        {
            'type': 'layered_agent',
            'call_id': '1004',
            'run': {
                'type': 'messages',
                'agent': 'bound_agent_2',
                'provider': 'MockProvider',
                'model': 'some_model',
                'model_kwargs': {'outer': False},
                'messages': [
                    {
                        'role': 'system',
                        'text': 'You are AI.',
                    },
                    {
                        'role': 'tool_call',
                        'tools': [
                            {
                                'call_id': '1004',
                                'call_type': 'function',
                                'function': {
                                    'arguments': '{"prompt": "Hey"}',
                                    'name': 'bound_agent_2',
                                },
                            },
                        ],
                    },
                    {
                        'role': 'ai',
                        'text': 'model: some_model',
                    },
                    {
                        'role': 'human',
                        'text': 'model_kwarg: outer = False',
                    },
                ],
            },
        },
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
    my_agent = build_simple_agent(name = 'agent')
    my_bound_agent = my_binder(my_agent)

    assert is_callable_of_type(my_agent, AgentCallable)
    assert is_callable_of_type(my_bound_agent, BoundAgentCallable)

    assert not is_callable_of_type(my_agent, BoundAgentCallable)
    assert not is_callable_of_type(my_bound_agent, AgentCallable)


def test_get_tool_params_function():
    def sample_tool(a: int, b: int, c: float) -> str:
        """
        Use this for
        anything you want.
        :param: a: int: first param
        :param: b: int: second param
        :param: c: float: (optional) another param
        """
        return ''
    description, params = get_tool_params(sample_tool)
    assert description == 'Use this for anything you want.'
    assert params == [
        {
            'name': 'a',
            'type': 'int',
            'description': 'first param',
        },
        {
            'name': 'b',
            'type': 'int',
            'description': 'second param',
        },
        {
            'name': 'c',
            'type': 'float',
            'description': '(optional) another param',
            'optional': True,
        },
    ]


def test_get_tool_params_layered_agents():
    my_binder = bind_model(MockProvider, 'some_model', {'a': 'yes', 'b': 6})
    my_agent = build_simple_agent(
        name = 'my_layered_agent',
        doc = 'Use this for everything.',
    )
    my_bound_agent = my_binder(my_agent)

    description, params = get_tool_params(my_agent)
    assert description == 'Use this for everything.'
    assert params == []

    description, params = get_tool_params(my_bound_agent)
    assert description == 'Use this for everything.'
    assert params == []

    my_binder = bind_model(MockProvider, 'some_model', {'a': 'yes', 'b': 6})
    my_agent = build_simple_agent(
        name = 'my_layered_agent',
        doc = """
        Use this for everything.
        :param: prompt: str: the prompt for the downstream AI
        """
    )
    my_bound_agent = my_binder(my_agent)

    description, params = get_tool_params(my_agent)
    assert description == 'Use this for everything.'
    assert params == [
        {
            'name': 'prompt',
            'type': 'str',
            'description': 'the prompt for the downstream AI',
        },
    ]

    description, params = get_tool_params(my_bound_agent)
    assert description == 'Use this for everything.'
    assert params == [
        {
            'name': 'prompt',
            'type': 'str',
            'description': 'the prompt for the downstream AI',
        },
    ]

    my_binder = bind_model(MockProvider, 'some_model', {'a': 'yes', 'b': 6})
    my_agent = build_simple_agent(
        name = 'my_layered_agent',
        doc = """
        Use this for everything.
        :param: prompt: int: a prompt as an int, weird
        """
    )
    my_bound_agent = my_binder(my_agent)

    description, params = get_tool_params(my_agent)
    assert description == 'Use this for everything.'
    assert params == [
        {
            'name': 'prompt',
            'type': 'int',
            'description': 'a prompt as an int, weird',
        },
    ]

    description, params = get_tool_params(my_bound_agent)
    assert description == 'Use this for everything.'
    assert params == [
        {
            'name': 'prompt',
            'type': 'int',
            'description': 'a prompt as an int, weird',
        },
    ]

    my_binder = bind_model(MockProvider, 'some_model', {'a': 'yes', 'b': 6})
    my_agent = build_simple_agent(
        name = 'my_layered_agent',
        doc = """
        Use this for everything.
        :param: prompt: str: the prompt
        :param: x: str: something else
        """
    )
    my_bound_agent = my_binder(my_agent)

    description, params = get_tool_params(my_agent)
    assert description == 'Use this for everything.'
    assert params == [
        {
            'name': 'prompt',
            'type': 'str',
            'description': 'the prompt',
        },
        {
            'name': 'x',
            'type': 'str',
            'description': 'something else',
        },
    ]

    description, params = get_tool_params(my_bound_agent)
    assert description == 'Use this for everything.'
    assert params == [
        {
            'name': 'prompt',
            'type': 'str',
            'description': 'the prompt',
        },
        {
            'name': 'x',
            'type': 'str',
            'description': 'something else',
        },
    ]


def test_extract_tool_result_as_sting():
    m1: ToolResult = {
        'call_id': '1001',
        'type': 'any',
        'result': 'something',
    }
    assert extract_tool_result_as_sting(m1) == 'something'

    m2: ToolResult = {
        'call_id': '1001',
        'type': 'layered_agent',
        'run': {
            'type': 'messages',
            'messages': [
                {
                    'role': 'ai',
                    'text': 'Hi',
                },
            ],
        },
    }
    assert extract_tool_result_as_sting(m2) == 'Hi'

    m3: ToolResult = {
        'call_id': '1001',
        'type': 'layered_agent',
        'run': {
            'type': 'messages',
            'messages': [
                {
                    'role': 'tool_call',
                    'tools': [
                        {
                            'call_id': '1002',
                            'call_type': 'function',
                            'function': {
                                'name': 'foo',
                                'arguments': '{"a": 7, "other": true}',
                            },
                        },
                    ],
                },
            ],
        },
    }
    assert extract_tool_result_as_sting(m3) == '[{"name": "foo", "values": {"a": 7, "other": true}}]'

    m4: ToolResult = {
        'call_id': '1001',
        'type': 'layered_agent',
        'run': {
            'type': 'messages',
            'messages': [
                {
                    'role': 'ai',
                    'text': 'Hi',
                },
                {
                    'role': 'ai',
                    'text': 'Hi again',
                },
            ],
        },
    }
    assert extract_tool_result_as_sting(m4) == 'Hi again'
