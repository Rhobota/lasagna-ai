import pytest
import copy

from lasagna.agent_util import (
    bind_model,
    build_simple_agent,
    flat_messages,
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
    Model,
    EventCallback,
    AgentRun,
)

from typing import List, Dict, Callable, Awaitable

from enum import Enum

from lasagna.tools_util import (
    convert_to_json_schema,
    get_tool_params,
    handle_tools,
    build_tool_response_message,
    is_async_callable,
    is_callable_of_type,
    extract_tool_result_as_sting,
    validate_args,
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


class Color(Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'


def tool_a(first, second, third=5):
    """
    Tool a
    :param: first: int: first param
    :param: second: int: second param
    :param: third: int: (optional) third param
    """
    return first + second + third

def tool_b(x):
    """
    Tool b
    :param: x: float: the value
    """
    return x * 2

async def tool_async_a(x):
    """
    Async tool a
    :param: x: float: the value
    """
    return x * 3

def tool_with_enum(c: Color):
    """
    A tool that accepts an enum.
    :param: c: enum red blue green: a param
    """
    return str(c)

def tool_with_enum_str_annotation(c: str):
    """
    A tool that accepts an enum.
    :param: c: enum red blue green: a param
    """
    return str(c)

def tool_with_enum_missing_annotation(c):
    """
    A tool that accepts an enum.
    :param: c: enum red blue green: a param
    """
    return str(c)

class ToolAsCallableObject:
    def __call__(self, a: int) -> int:
        """
        This is a callable object tool.
        :param: a: int: the param named `a`
        """
        return 4 * a

class ToolAsAsyncCallableObject:
    async def __call__(self, a: int) -> int:
        """
        This is an async callable object tool.
        :param: a: int: the param named `a`
        """
        return 5 * a

async def agent_as_function(
    model: Model,
    event_callback: EventCallback,
    prev_runs: List[AgentRun],
) -> AgentRun:
    """This is an agent as a function."""
    return flat_messages('agent_as_function', [])

class AgentAsAsyncCallableObject:
    async def __call__(
        self,
        model: Model,
        event_callback: EventCallback,
        prev_runs: List[AgentRun],
    ) -> AgentRun:
        """This is an agent as an async callable object."""
        return flat_messages('AgentAsAsyncCallableObject', [])


def test_validate_args():
    with pytest.raises(TypeError) as e:
        validate_args(
            tool_a,
            {
                'second': 4,
                'first': 1,
                'third': 2,
                'other': 5,
            },
        )
    assert str(e.value) == "tool_a() got 1 unexpected argument: 'other'"

    with pytest.raises(TypeError) as e:
        validate_args(
            tool_a,
            {
                'second': 4,
                'first': 1,
                'third': 2,
                'other_1': 5,
                'other_2': 7,
            },
        )
    assert str(e.value) == "tool_a() got 2 unexpected arguments: 'other_1', 'other_2'"

    with pytest.raises(TypeError) as e:
        validate_args(
            tool_a,
            {
                'other_1': 5,
                'other_2': 7,
            },
        )
    assert str(e.value) == "tool_a() got 2 unexpected arguments: 'other_1', 'other_2'"

    with pytest.raises(TypeError) as e:
        validate_args(
            tool_a,
            {},
        )
    assert str(e.value) == "tool_a() missing 2 required arguments: 'first', 'second'"

    with pytest.raises(TypeError) as e:
        validate_args(
            tool_a,
            {'third': 6},
        )
    assert str(e.value) == "tool_a() missing 2 required arguments: 'first', 'second'"

    with pytest.raises(TypeError) as e:
        validate_args(
            tool_a,
            {'second': 5},
        )
    assert str(e.value) == "tool_a() missing 1 required argument: 'first'"

    with pytest.raises(TypeError) as e:
        validate_args(
            tool_a,
            {'first': 5},
        )
    assert str(e.value) == "tool_a() missing 1 required argument: 'second'"

    with pytest.raises(TypeError) as e:
        validate_args(
            tool_a,
            {
                'second': 'hi',
                'first': 1,
            },
        )
    assert str(e.value) == "tool_a() got invalid value for argument `second`: 'hi' (invalid literal for int() with base 10: 'hi')"

    with pytest.raises(TypeError) as e:
        validate_args(
            tool_a,
            {
                'second': 1,
                'first': 'hi',
            },
        )
    assert str(e.value) == "tool_a() got invalid value for argument `first`: 'hi' (invalid literal for int() with base 10: 'hi')"

    args = validate_args(
        tool_a,
        {
            'second': 1.8,
            'first': 2,
        },
    )
    assert args == {
        'second': 1,
        'first': 2,
    }

    args = validate_args(
        tool_a,
        {
            'second': 3.2,
            'first': 2,
            'third': True,
        },
    )
    assert args == {
        'second': 3,
        'first': 2,
        'third': 1,
    }

    with pytest.raises(TypeError) as e:
        validate_args(
            tool_with_enum,
            {
                'c': 1,
            },
        )
    assert str(e.value) == "tool_with_enum() got invalid value for argument `c`: 1 (valid values are ['blue', 'green', 'red'])"

    with pytest.raises(TypeError) as e:
        validate_args(
            tool_with_enum,
            {
                'c': 'yellow',
            },
        )
    assert str(e.value) == "tool_with_enum() got invalid value for argument `c`: 'yellow' (valid values are ['blue', 'green', 'red'])"

    args = validate_args(
        tool_with_enum,
        {
            'c': 'red',
        },
    )
    assert args == {'c': Color.RED}

    args = validate_args(
        tool_with_enum_str_annotation,
        {
            'c': 'red',
        },
    )
    assert args == {'c': 'red'}

    args = validate_args(
        tool_with_enum_missing_annotation,
        {
            'c': 'red',
        },
    )
    assert args == {'c': 'red'}

    args = validate_args(
        ToolAsCallableObject(),
        {
            'a': 5,
        },
    )
    assert args == {'a': 5}

    args = validate_args(
        ToolAsAsyncCallableObject(),
        {
            'a': 5,
        },
    )
    assert args == {'a': 5}

    args = validate_args(
        agent_as_function,
        {
        },
    )
    assert args == {}

    args = validate_args(
        AgentAsAsyncCallableObject(),
        {
        },
    )
    assert args == {}


@pytest.mark.asyncio
async def test_handle_tools_standard_functions():
    x = 4
    def tool_c():
        """
        Tool c
        """
        return x * 4

    async def tool_async_b():
        """
        Async tool b
        """
        return x * 5

    tool_map: Dict[str, Callable] = {
        'tool_a': tool_a,
        'tool_b': tool_b,
        'tool_c': tool_c,
        'tool_object': ToolAsCallableObject(),
        'tool_async_a': tool_async_a,
        'tool_async_b': tool_async_b,
        'tool_async_object': ToolAsAsyncCallableObject(),
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
            {'call_id': '1017', 'function': {'arguments': '[1, 2, 3]', 'name': 'tool_a'}, 'call_type': 'function'},
            {'call_id': '1018', 'function': {'arguments': '{"a": 5}', 'name': 'tool_object'}, 'call_type': 'function'},
            {'call_id': '1019', 'function': {'arguments': '{"a": 5}', 'name': 'tool_async_object'}, 'call_type': 'function'},
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
        {'type': 'any', 'call_id': '1001', 'result': 16.0 },
        {'type': 'any', 'call_id': '1002', 'result': 10.8 },
        {'type': 'any', 'call_id': '1003', 'result': "TypeError: tool_b() got invalid value for argument `x`: 'hi' (could not convert string to float: 'hi')", 'is_error': True, },
        {'type': 'any', 'call_id': '1004', 'result': "TypeError: tool_b() missing 1 required argument: 'x'", 'is_error': True },
        {'type': 'any', 'call_id': '1005', 'result': "TypeError: tool_b() got 1 unexpected argument: 'y'", 'is_error': True },
        {'type': 'any', 'call_id': '1006', 'result': "TypeError: tool_b() got 1 unexpected argument: 'y'", 'is_error': True },
        {'type': 'any', 'call_id': '1007', 'result': 17 },
        {'type': 'any', 'call_id': '1008', 'result': 15 },
        {'type': 'any', 'call_id': '1009', 'result': 111 },
        {'type': 'any', 'call_id': '1010', 'result': "TypeError: tool_a() missing 2 required arguments: 'first', 'second'", 'is_error': True },
        {'type': 'any', 'call_id': '1011', 'result': "TypeError: tool_a() missing 1 required argument: 'second'", 'is_error': True },
        {'type': 'any', 'call_id': '1012', 'result': 16 },
        {'type': 'any', 'call_id': '1013', 'result': "KeyError: 'tool_d'", 'is_error': True },
        {'type': 'any', 'call_id': '1014', 'result': -9.0 },
        {'type': 'any', 'call_id': '1015', 'result': 20 },
        {'type': 'any', 'call_id': '1016', 'result': "TypeError: tool_async_a() missing 1 required argument: 'x'", 'is_error': True },
        {'type': 'any', 'call_id': '1017', 'result': "TypeError: tool output must be a JSON object", 'is_error': True },
        {'type': 'any', 'call_id': '1018', 'result': 20 },
        {'type': 'any', 'call_id': '1019', 'result': 25 },
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

    doc = """
    An agent.
    :param: prompt: str: (optional) the prompt
    """

    agent_1 = build_simple_agent(
        name = 'agent_1',
        doc = doc,
    )
    my_binder = bind_model(**inner_model_spec)
    bound_agent_2 = my_binder(
        build_simple_agent(
            name = 'bound_agent_2',
            doc = doc,
        ),
    )

    tool_map: Dict[str, Callable] = {
        'agent_1': agent_1,
        'bound_agent_2': bound_agent_2,
        'agent_as_function': agent_as_function,
        'agent_as_callable_object': AgentAsAsyncCallableObject(),
    }

    message: Message = {
        'role': 'tool_call',
        'tools': [
            {'call_id': '1001', 'function': {'arguments': '{}', 'name': 'agent_1'}, 'call_type': 'function'},
            {'call_id': '1002', 'function': {'arguments': '{}', 'name': 'bound_agent_2'}, 'call_type': 'function'},
            {'call_id': '1003', 'function': {'arguments': '{"prompt": "Hi"}', 'name': 'agent_1'}, 'call_type': 'function'},
            {'call_id': '1004', 'function': {'arguments': '{"prompt": "Hey"}', 'name': 'bound_agent_2'}, 'call_type': 'function'},
            {'call_id': '1005', 'function': {'arguments': '{"bad_name": "Hi"}', 'name': 'agent_1'}, 'call_type': 'function'},
            {'call_id': '1006', 'function': {'arguments': '{"bad_name": "Hey"}', 'name': 'bound_agent_2'}, 'call_type': 'function'},
            {'call_id': '1007', 'function': {'arguments': '{}', 'name': 'agent_as_function'}, 'call_type': 'function'},
            {'call_id': '1008', 'function': {'arguments': '{}', 'name': 'agent_as_callable_object'}, 'call_type': 'function'},
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
        {
            'type': 'any',
            'call_id': '1005',
            'is_error': True,
            'result': "TypeError: agent_1() got 1 unexpected argument: 'bad_name'",
        },
        {
            'type': 'any',
            'call_id': '1006',
            'is_error': True,
            'result': "TypeError: bound_agent_2() got 1 unexpected argument: 'bad_name'",
        },
        {
            'type': 'layered_agent',
            'call_id': '1007',
            'run': {
                'type': 'messages',
                'agent': 'agent_as_function',
                'provider': 'MockProvider',
                'model': 'some_model',
                'model_kwargs': {'outer': True},
                'messages': [],
            },
        },
        {
            'type': 'layered_agent',
            'call_id': '1008',
            'run': {
                'type': 'messages',
                'agent': 'AgentAsAsyncCallableObject',
                'provider': 'MockProvider',
                'model': 'some_model',
                'model_kwargs': {'outer': True},
                'messages': [],
            },
        },
    ]


def test_build_tool_response_message():
    res: List[ToolResult] = [
        {'type': 'any', 'call_id': '1002', 'result': 10.8 },
        {'type': 'any', 'call_id': '1003', 'result': "hihi" },
    ]
    message = build_tool_response_message(copy.deepcopy(res))
    assert message == {
        'role': 'tool_res',
        'tools': res,
    }


MyCallable = Callable[[int, str], bool]
MyAsyncCallable = Callable[[int, str], Awaitable[bool]]

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

    assert not is_async_callable(ToolAsCallableObject())
    assert is_async_callable(ToolAsAsyncCallableObject())

    assert is_async_callable(agent_as_function)
    assert is_async_callable(AgentAsAsyncCallableObject())


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
    assert matching_objects[0] is not_async
    assert isinstance(matching_objects[1], not_async_class)  # !!!

    matching_objects = [
        o
        for o in objects
        if is_callable_of_type(o, MyAsyncCallable, no_throw=True)
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

    assert not is_callable_of_type(ToolAsCallableObject(), AgentCallable)
    assert not is_callable_of_type(ToolAsAsyncCallableObject(), AgentCallable)

    assert is_callable_of_type(agent_as_function, AgentCallable)
    assert is_callable_of_type(AgentAsAsyncCallableObject(), AgentCallable)


def test_get_tool_params_function():
    def sample_tool(
        a: int,
        b: str,
        c: bool,
        d: Color, # <-- we support enums as Enum
        e: str,   # <-- we support enums as str
        f: float = 0.0,
    ) -> str:
        """
        Use this for
        anything you want.
        :param: a: int: first param
        :param: b: str: second param
        :param: c: bool: third param
        :param: d: enum red blue green: forth param
        :param: e: enum red blue green: fifth param
        :param: f: float: (optional) last param
        """
        return ''
    description, params = get_tool_params(sample_tool)
    assert description == 'Use this for\nanything you want.'
    assert params == [
        {
            'name': 'a',
            'type': 'int',
            'description': 'first param',
        },
        {
            'name': 'b',
            'type': 'str',
            'description': 'second param',
        },
        {
            'name': 'c',
            'type': 'bool',
            'description': 'third param',
        },
        {
            'name': 'd',
            'type': 'enum red blue green',
            'description': 'forth param',
        },
        {
            'name': 'e',
            'type': 'enum red blue green',
            'description': 'fifth param',
        },
        {
            'name': 'f',
            'type': 'float',
            'description': '(optional) last param',
            'optional': True,
        },
    ]

    def param_count_mismatch(a: int, b: str) -> str:
        """
        A function
        :param: a: int: first param
        """
        return ''
    with pytest.raises(ValueError) as e:
        get_tool_params(param_count_mismatch)
    assert str(e.value) == 'tool `param_count_mismatch` has parameter length mismatch: tool has 2, docstring has 1'

    def name_mismatch(a: int) -> str:
        """
        A function
        :param: b: int: first param
        """
        return ''
    with pytest.raises(ValueError) as e:
        get_tool_params(name_mismatch)
    assert str(e.value) == 'tool `name_mismatch` has parameter name mismatch: tool name is `a`, docstring name is `b`'

    def no_default_on_optional(a: int) -> str:
        """
        A function
        :param: a: int: (optional) first param
        """
        return ''
    with pytest.raises(ValueError) as e:
        get_tool_params(no_default_on_optional)
    assert str(e.value) == 'tool `no_default_on_optional` has an optional parameter without a default value: `a`'

    def enum_type_mismatch_1(a: Color) -> str:
        """
        A function
        :param: a: int: first param
        """
        return ''
    with pytest.raises(ValueError) as e:
        get_tool_params(enum_type_mismatch_1)
    assert str(e.value) == "tool `enum_type_mismatch_1` has parameter `a` type mismatch: tool type is `<enum 'Color'>`, docstring type is `<class 'int'>`"

    def enum_type_mismatch_2(a: int) -> str:
        """
        A function
        :param: a: enum red green blue: first param
        """
        return ''
    with pytest.raises(ValueError) as e:
        get_tool_params(enum_type_mismatch_2)
    assert str(e.value) == "tool `enum_type_mismatch_2` has parameter `a` type mismatch: tool type is `<class 'int'>`, docstring type is `enum red green blue`"

    def enum_type_mismatch_3(a: Color) -> str:
        """
        A function
        :param: a: enum red GREEN blue: first param
        """
        return ''
    with pytest.raises(ValueError) as e:
        get_tool_params(enum_type_mismatch_3)
    assert str(e.value) == "tool `enum_type_mismatch_3` has parameter `a` enum value mismatch: tool has enum values `['blue', 'green', 'red']`, docstring has enum values `['GREEN', 'blue', 'red']`"

    def type_mismatch(a: int) -> str:
        """
        A function
        :param: a: float: first param
        """
        return ''
    with pytest.raises(ValueError) as e:
        get_tool_params(type_mismatch)
    assert str(e.value) == "tool `type_mismatch` has parameter `a` type mismatch: tool type is `<class 'int'>`, docstring type is `<class 'float'>`"

    def unknown_type(a: Callable) -> str:
        """
        A function
        :param: a: float: first param
        """
        return ''
    with pytest.raises(ValueError) as e:
        get_tool_params(unknown_type)
    assert str(e.value) == "tool `unknown_type` has parameter `a` type mismatch: tool type is `typing.Callable`, docstring type is `<class 'float'>`"


def test_get_tool_params_async_function():
    async def sample_tool(
        a: int,
        b: str,
        c: bool,
        d: Color, # <-- we support enums as Enum
        e: str,   # <-- we support enums as str
        f: float = 0.0,
    ) -> str:
        """
        Use this for
        anything you want.
        :param: a: int: first param
        :param: b: str: second param
        :param: c: bool: third param
        :param: d: enum red blue green: forth param
        :param: e: enum red blue green: fifth param
        :param: f: float: (optional) last param
        """
        return ''
    description, params = get_tool_params(sample_tool)
    assert description == 'Use this for\nanything you want.'
    assert params == [
        {
            'name': 'a',
            'type': 'int',
            'description': 'first param',
        },
        {
            'name': 'b',
            'type': 'str',
            'description': 'second param',
        },
        {
            'name': 'c',
            'type': 'bool',
            'description': 'third param',
        },
        {
            'name': 'd',
            'type': 'enum red blue green',
            'description': 'forth param',
        },
        {
            'name': 'e',
            'type': 'enum red blue green',
            'description': 'fifth param',
        },
        {
            'name': 'f',
            'type': 'float',
            'description': '(optional) last param',
            'optional': True,
        },
    ]


def test_get_tool_params_callable_object():
    class sample_tool:
        def __call__(
            self,
            a: int,
            b: str,
            c: bool,
            d: Color, # <-- we support enums as Enum
            e: str,   # <-- we support enums as str
            f: float = 0.0,
        ) -> str:
            """
            Use this for
            anything you want.
            :param: a: int: first param
            :param: b: str: second param
            :param: c: bool: third param
            :param: d: enum red blue green: forth param
            :param: e: enum red blue green: fifth param
            :param: f: float: (optional) last param
            """
            return ''
    description, params = get_tool_params(sample_tool())
    assert description == 'Use this for\nanything you want.'
    assert params == [
        {
            'name': 'a',
            'type': 'int',
            'description': 'first param',
        },
        {
            'name': 'b',
            'type': 'str',
            'description': 'second param',
        },
        {
            'name': 'c',
            'type': 'bool',
            'description': 'third param',
        },
        {
            'name': 'd',
            'type': 'enum red blue green',
            'description': 'forth param',
        },
        {
            'name': 'e',
            'type': 'enum red blue green',
            'description': 'fifth param',
        },
        {
            'name': 'f',
            'type': 'float',
            'description': '(optional) last param',
            'optional': True,
        },
    ]


def test_get_tool_params_async_callable_object():
    class sample_tool:
        async def __call__(
            self,
            a: int,
            b: str,
            c: bool,
            d: Color, # <-- we support enums as Enum
            e: str,   # <-- we support enums as str
            f: float = 0.0,
        ) -> str:
            """
            Use this for
            anything you want.
            :param: a: int: first param
            :param: b: str: second param
            :param: c: bool: third param
            :param: d: enum red blue green: forth param
            :param: e: enum red blue green: fifth param
            :param: f: float: (optional) last param
            """
            return ''
    description, params = get_tool_params(sample_tool())
    assert description == 'Use this for\nanything you want.'
    assert params == [
        {
            'name': 'a',
            'type': 'int',
            'description': 'first param',
        },
        {
            'name': 'b',
            'type': 'str',
            'description': 'second param',
        },
        {
            'name': 'c',
            'type': 'bool',
            'description': 'third param',
        },
        {
            'name': 'd',
            'type': 'enum red blue green',
            'description': 'forth param',
        },
        {
            'name': 'e',
            'type': 'enum red blue green',
            'description': 'fifth param',
        },
        {
            'name': 'f',
            'type': 'float',
            'description': '(optional) last param',
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
            'agent': 'some_downstream_agent',
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
            'agent': 'some_downstream_agent',
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
            'agent': 'some_downstream_agent',
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
