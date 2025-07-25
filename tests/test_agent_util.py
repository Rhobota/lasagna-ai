import pytest

from typing import Callable, List

from lasagna.agent_util import (
    bind_model,
    build_simple_agent,
    build_standard_message_extractor,
    extract_last_message,
    noop_callback,
    partial_bind_model,
    recursive_extract_messages,
    recursive_sum_costs,
    flat_messages,
    build_extraction_agent,
    override_system_prompt,
)

from lasagna.types import (
    AgentCallable,
    BoundAgentCallable,
    AgentRun,
    Message,
    EventPayload,
)

from lasagna.mock_provider import (
    MockProvider,
)

from lasagna.util import get_name

from pydantic import BaseModel, ValidationError


_AGENT_RUN: AgentRun = {
    'agent': 'outer_agent',
    'type': 'chain',
    'runs': [
        {
            'agent': 'inner_agent_1',
            'type': 'parallel',
            'runs': [
                {
                    'agent': 'inner_agent_2',
                    'type': 'messages',
                    'messages': [
                        {
                            'role': 'system',
                            'text': 'You are a robot.',
                        },
                        {
                            'role': 'human',
                            'text': 'What are you?',
                        },
                    ],
                },
                {
                    'agent': 'inner_agent_3',
                    'type': 'messages',
                    'messages': [
                        {
                            'role': 'system',
                            'text': 'You are a cat.',
                        },
                        {
                            'role': 'human',
                            'text': 'Here kitty kitty!',
                        },
                        {
                            'role': 'tool_res',
                            'tools': [
                                {
                                    'type': 'any',
                                    'call_id': 'call000',
                                    'result': 'Meow.',
                                },
                                {
                                    'type': 'layered_agent',
                                    'call_id': 'call001',
                                    'run': {
                                        'agent': 'inner_agent_4',
                                        'type': 'messages',
                                        'messages': [
                                            {
                                                'role': 'ai',
                                                'text': 'Beep.',
                                                'cost': {
                                                    'input_tokens': 10,
                                                    'output_tokens': 2,
                                                    'total_tokens': 12,
                                                },
                                            },
                                        ],
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
        },
        {
            'agent': 'inner_agent_5',
            'type': 'extraction',
            'messages': [
                {
                    'role': 'tool_call',
                    'tools': [
                        {
                            'call_id': 'call002',
                            'call_type': 'function',
                            'function': {
                                'name': 'foo',
                                'arguments': '{"value": 7}',
                            },
                        },
                    ],
                },
            ],
            'result': {'value': 7},
        },
        {
            'agent': 'inner_agent_6',
            'type': 'messages',
            'messages': [
                {
                    'role': 'system',
                    'text': 'You aggregate other AI systems.',
                    'cost': {
                        'output_tokens': 3,
                    },
                },
                {
                    'role': 'human',
                    'text': 'Summarize the previous AI conversations, please.',
                },
            ],
        },
    ],
}


class MyTestType(BaseModel):
    a: str
    b: int


async def _agent_common_test(
    binder: Callable[[AgentCallable], BoundAgentCallable],
    agent: AgentCallable,
):
    events = []
    async def event_callback(event: EventPayload) -> None:
        events.append(event)
    prev_runs: List[AgentRun] = []
    new_run = await binder(agent)(event_callback, prev_runs)
    assert new_run == {
        'agent': 'simple_agent',
        'provider': 'MockProvider',
        'model': 'some_model',
        'model_kwargs': {
            'b': 6,
            'a': 'yes',
        },
        'type': 'messages',
        'messages': [
            {
                'role': 'ai',
                'text': f"model: some_model",
            },
            {
                'role': 'human',
                'text': f"model_kwarg: a = yes",
            },
            {
                'role': 'human',
                'text': f"model_kwarg: b = 6",
            },
        ],
    }
    assert events == [
        ('agent', 'start', 'simple_agent'),
        ('ai', 'text_event', 'Hi!'),
        ('agent', 'end', new_run),
    ]


@pytest.mark.asyncio
async def test_bind_model():
    my_binder = bind_model(MockProvider, 'some_model', a = 'yes', b = 6)
    my_agent = build_simple_agent(name = 'simple_agent')
    await _agent_common_test(my_binder, my_agent)


@pytest.mark.asyncio
async def test_partial_bind_model():
    my_binder = partial_bind_model(MockProvider, 'some_model')(a = 'yes', b = 6)
    my_agent = build_simple_agent(name = 'simple_agent')
    await _agent_common_test(my_binder, my_agent)


@pytest.mark.asyncio
async def test_build_layered_agent():
    my_binder = bind_model(MockProvider, 'a_model')
    my_agent = build_simple_agent(
        name = 'a_layered_agent',
        tools = [],
        message_extractor = build_standard_message_extractor(
            system_prompt_override = 'system test',
        ),
        doc = 'doc test',
    )
    assert my_agent.__doc__ == 'doc test'
    assert str(my_agent) == 'a_layered_agent'
    assert get_name(my_agent) == 'a_layered_agent'
    my_bound_agent = my_binder(my_agent)
    assert my_bound_agent.__doc__ == 'doc test'
    assert my_bound_agent.__name__ == 'a_layered_agent'
    assert get_name(my_bound_agent) == 'a_layered_agent'
    prev_runs: List[AgentRun] = [
        flat_messages(
            'some_agent',
            [
                {
                    'role': 'human',
                    'text': 'layered agent test',
                },
            ],
        ),
    ]
    new_run = await my_bound_agent(noop_callback, prev_runs)
    assert new_run == {
        'agent': 'a_layered_agent',
        'provider': 'MockProvider',
        'model': 'a_model',
        'model_kwargs': {},
        'type': 'messages',
        'messages': [
            {
                'role': 'system',
                'text': 'system test',
            },
            {
                'role': 'human',
                'text': 'layered agent test',
            },
            {
                'role': 'ai',
                'text': f"model: a_model",
            },
        ],
    }


@pytest.mark.asyncio
async def test_model_extract():
    my_binder = bind_model(MockProvider, 'some_model', a = 'yes', b = 6)
    events = []
    async def event_callback(event: EventPayload) -> None:
        events.append(event)
    prev_runs: List[AgentRun] = []
    run = await my_binder(
        build_extraction_agent(
            name = 'my_extraction_agent',
            extraction_type = MyTestType,
        ),
    )(event_callback, prev_runs)
    assert events == [
        (
            'agent',
            'start',
            'my_extraction_agent',
        ),
        (
            'tool_call',
            'tool_call_event',
            {
                'call_id': 'id123',
                'call_type': 'function',
                'function': {
                    'name': 'f',
                    'arguments': '{"a": "yes", "b": 6}',
                },
            },
        ),
        (
            'agent',
            'end',
            {
                'type': 'extraction',
                'messages': [
                    {
                        'role': 'ai',
                        'text': None,
                    },
                ],
                'result': MyTestType(a='yes', b=6),
                'provider': 'MockProvider',
                'agent': 'my_extraction_agent',
                'model': 'some_model',
                'model_kwargs': {
                    'a': 'yes',
                    'b': 6,
                },
            },
        ),
    ]
    assert run['type'] == 'extraction'
    assert run['messages'] == [
        {
            'role': 'ai',
            'text': None,
        },
    ]
    result = run['result']
    assert isinstance(result, MyTestType)
    assert result.a == 'yes'
    assert result.b == 6


@pytest.mark.asyncio
async def test_model_extract_type_mismatch():
    my_binder = bind_model(MockProvider, 'some_model', a = 'yes', b = 'BAD VALUE')
    prev_runs: List[AgentRun] = []
    with pytest.raises(ValidationError):
        await my_binder(
            build_extraction_agent(
                name = 'my_extraction_agent',
                extraction_type = MyTestType,
            ),
        )(noop_callback, prev_runs)


def test_recursive_extract_messages():
    assert recursive_extract_messages(
        _AGENT_RUN,
        from_tools=True,
        from_extraction=True,
    ) == [
        {
            'role': 'system',
            'text': 'You are a robot.',
        },
        {
            'role': 'human',
            'text': 'What are you?',
        },
        {
            'role': 'system',
            'text': 'You are a cat.',
        },
        {
            'role': 'human',
            'text': 'Here kitty kitty!',
        },
        {
            'role': 'tool_res',
            'tools': [
                {
                    'type': 'any',
                    'call_id': 'call000',
                    'result': 'Meow.',
                },
                {
                    'type': 'layered_agent',
                    'call_id': 'call001',
                    'run': {
                        'agent': 'inner_agent_4',
                        'type': 'messages',
                        'messages': [
                            {
                                'role': 'ai',
                                'text': 'Beep.',
                                'cost': {
                                    'input_tokens': 10,
                                    'output_tokens': 2,
                                    'total_tokens': 12,
                                },
                            },
                        ],
                    },
                },
            ],
        },
        {
            'role': 'ai',
            'text': 'Beep.',
            'cost': {
                'input_tokens': 10,
                'output_tokens': 2,
                'total_tokens': 12,
            },
        },
        {
            'role': 'tool_call',
            'tools': [
                {
                    'call_id': 'call002',
                    'call_type': 'function',
                    'function': {
                        'name': 'foo',
                        'arguments': '{"value": 7}',
                    },
                },
            ],
        },
        {
            'role': 'system',
            'text': 'You aggregate other AI systems.',
            'cost': {
                'output_tokens': 3,
            },
        },
        {
            'role': 'human',
            'text': 'Summarize the previous AI conversations, please.',
        },
    ]
    assert recursive_extract_messages(
        _AGENT_RUN,
        from_tools=False,
        from_extraction=True,
    ) == [
        {
            'role': 'system',
            'text': 'You are a robot.',
        },
        {
            'role': 'human',
            'text': 'What are you?',
        },
        {
            'role': 'system',
            'text': 'You are a cat.',
        },
        {
            'role': 'human',
            'text': 'Here kitty kitty!',
        },
        {
            'role': 'tool_res',
            'tools': [
                {
                    'type': 'any',
                    'call_id': 'call000',
                    'result': 'Meow.',
                },
                {
                    'type': 'layered_agent',
                    'call_id': 'call001',
                    'run': {
                        'agent': 'inner_agent_4',
                        'type': 'messages',
                        'messages': [
                            {
                                'role': 'ai',
                                'text': 'Beep.',
                                'cost': {
                                    'input_tokens': 10,
                                    'output_tokens': 2,
                                    'total_tokens': 12,
                                },
                            },
                        ],
                    },
                },
            ],
        },
        {
            'role': 'tool_call',
            'tools': [
                {
                    'call_id': 'call002',
                    'call_type': 'function',
                    'function': {
                        'name': 'foo',
                        'arguments': '{"value": 7}',
                    },
                },
            ],
        },
        {
            'role': 'system',
            'text': 'You aggregate other AI systems.',
            'cost': {
                'output_tokens': 3,
            },
        },
        {
            'role': 'human',
            'text': 'Summarize the previous AI conversations, please.',
        },
    ]
    assert recursive_extract_messages(
        _AGENT_RUN,
        from_tools=False,
        from_extraction=False,
    ) == [
        {
            'role': 'system',
            'text': 'You are a robot.',
        },
        {
            'role': 'human',
            'text': 'What are you?',
        },
        {
            'role': 'system',
            'text': 'You are a cat.',
        },
        {
            'role': 'human',
            'text': 'Here kitty kitty!',
        },
        {
            'role': 'tool_res',
            'tools': [
                {
                    'type': 'any',
                    'call_id': 'call000',
                    'result': 'Meow.',
                },
                {
                    'type': 'layered_agent',
                    'call_id': 'call001',
                    'run': {
                        'agent': 'inner_agent_4',
                        'type': 'messages',
                        'messages': [
                            {
                                'role': 'ai',
                                'text': 'Beep.',
                                'cost': {
                                    'input_tokens': 10,
                                    'output_tokens': 2,
                                    'total_tokens': 12,
                                },
                            },
                        ],
                    },
                },
            ],
        },
        {
            'role': 'system',
            'text': 'You aggregate other AI systems.',
            'cost': {
                'output_tokens': 3,
            },
        },
        {
            'role': 'human',
            'text': 'Summarize the previous AI conversations, please.',
        },
    ]
    assert extract_last_message(_AGENT_RUN, from_tools=True, from_extraction=True) == {
        'role': 'human',
        'text': 'Summarize the previous AI conversations, please.',
    }


def test_flat_messages():
    messages: List[Message] = [
        {
            'role': 'system',
            'text': 'You are a cat.',
        },
        {
            'role': 'human',
            'text': 'Here kitty kitty!',
        },
    ]
    assert flat_messages('an_agent', messages) == {
        'agent': 'an_agent',
        'type': 'messages',
        'messages': [
            {
                'role': 'system',
                'text': 'You are a cat.',
            },
            {
                'role': 'human',
                'text': 'Here kitty kitty!',
            },
        ],
    }


def test_override_system_prompt():
    messages: List[Message] = []
    assert override_system_prompt(messages, 'Bla') == [
        {
            'role': 'system',
            'text': 'Bla',
        },
    ]
    assert messages == []  # test immutability

    messages: List[Message] = [
        {
            'role': 'system',
            'text': 'You are a robot.',
        },
    ]
    assert override_system_prompt(messages, 'You are Fred.') == [
        {
            'role': 'system',
            'text': 'You are Fred.',
        },
    ]
    assert messages == [   # test immutability
        {
            'role': 'system',
            'text': 'You are a robot.',
        },
    ]

    messages: List[Message] = [
        {
            'role': 'human',
            'text': 'Hi!',
        },
    ]
    assert override_system_prompt(messages, 'You are Fred.') == [
        {
            'role': 'system',
            'text': 'You are Fred.',
        },
        {
            'role': 'human',
            'text': 'Hi!',
        },
    ]
    assert messages == [   # test immutability
        {
            'role': 'human',
            'text': 'Hi!',
        },
    ]

    messages: List[Message] = [
        {
            'role': 'system',
            'text': 'You are a robot.',
        },
        {
            'role': 'human',
            'text': 'Who are you?',
        },
    ]
    assert override_system_prompt(messages, 'You are Fred.') == [
        {
            'role': 'system',
            'text': 'You are Fred.',
        },
        {
            'role': 'human',
            'text': 'Who are you?',
        },
    ]
    assert messages == [   # test immutability
        {
            'role': 'system',
            'text': 'You are a robot.',
        },
        {
            'role': 'human',
            'text': 'Who are you?',
        },
    ]

    messages: List[Message] = [
        {
            'role': 'ai',
            'text': 'Beep.',
        },
        {
            'role': 'human',
            'text': 'Who are you?',
        },
    ]
    assert override_system_prompt(messages, 'You are Fred.') == [
        {
            'role': 'system',
            'text': 'You are Fred.',
        },
        {
            'role': 'ai',
            'text': 'Beep.',
        },
        {
            'role': 'human',
            'text': 'Who are you?',
        },
    ]
    assert messages == [   # test immutability
        {
            'role': 'ai',
            'text': 'Beep.',
        },
        {
            'role': 'human',
            'text': 'Who are you?',
        },
    ]


def test_recursive_sum_costs():
    assert recursive_sum_costs(_AGENT_RUN) == {
        'input_tokens': 10,
        'output_tokens': 5,
        'total_tokens': 12,
    }
