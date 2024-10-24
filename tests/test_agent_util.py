import pytest

from typing import Callable, List

from lasagna.agent_util import (
    bind_model,
    build_most_simple_agent,
    extract_last_message,
    noop_callback,
    partial_bind_model,
    recursive_extract_messages,
    flat_messages,
    build_extraction_agent,
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

from pydantic import BaseModel, ValidationError


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
        'agent': 'simple agent',
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
                'cost': None,
                'raw': None,
            },
            {
                'role': 'human',
                'text': f"model_kwarg: a = yes",
                'cost': None,
                'raw': None,
            },
            {
                'role': 'human',
                'text': f"model_kwarg: b = 6",
                'cost': None,
                'raw': None,
            },
        ],
    }
    assert events == [
        ('agent', 'start', 'simple agent'),
        ('ai', 'text_event', 'Hi!'),
        ('agent', 'end', new_run),
    ]


@pytest.mark.asyncio
async def test_bind_model():
    my_binder = bind_model(MockProvider, 'some_model', {'a': 'yes', 'b': 6})
    my_agent = build_most_simple_agent()
    await _agent_common_test(my_binder, my_agent)


@pytest.mark.asyncio
async def test_partial_bind_model():
    my_binder = partial_bind_model(MockProvider, 'some_model')({'a': 'yes', 'b': 6})
    my_agent = build_most_simple_agent()
    await _agent_common_test(my_binder, my_agent)


@pytest.mark.asyncio
async def test_model_extract():
    my_binder = bind_model(MockProvider, 'some_model', {'a': 'yes', 'b': 6})
    events = []
    async def event_callback(event: EventPayload) -> None:
        events.append(event)
    prev_runs: List[AgentRun] = []
    run = await my_binder(build_extraction_agent(MyTestType))(event_callback, prev_runs)
    assert events == [
        (
            'agent',
            'start',
            'extraction agent: MyTestType',
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
                'message': {
                    'role': 'ai',
                    'text': None,
                },
                'result': MyTestType(a='yes', b=6),
                'provider': 'MockProvider',
                'agent': 'extraction agent: MyTestType',
                'model': 'some_model',
                'model_kwargs': {
                    'a': 'yes',
                    'b': 6,
                },
            },
        ),
    ]
    assert run['type'] == 'extraction'
    assert run['message'] == {
        'role': 'ai',
        'text': None,
    }
    result = run['result']
    assert isinstance(result, MyTestType)
    assert result.a == 'yes'
    assert result.b == 6


@pytest.mark.asyncio
async def test_model_extract_type_mismatch():
    my_binder = bind_model(MockProvider, 'some_model', {'a': 'yes', 'b': 'BAD VALUE'})
    prev_runs: List[AgentRun] = []
    with pytest.raises(ValidationError):
        await my_binder(build_extraction_agent(MyTestType))(noop_callback, prev_runs)


def test_recursive_extract_messages():
    agent_run: AgentRun = {
        'type': 'chain',
        'runs': [
            {
                'type': 'parallel',
                'runs': [
                    {
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
                    },
                ],
            },
            {
                'type': 'messages',
                'messages': [
                    {
                        'role': 'system',
                        'text': 'You aggregate other AI systems.',
                    },
                    {
                        'role': 'human',
                        'text': 'Summarize the previous AI conversations, please.',
                    },
                ],
            },
        ],
    }
    assert recursive_extract_messages([agent_run]) == [
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
            'role': 'system',
            'text': 'You aggregate other AI systems.',
        },
        {
            'role': 'human',
            'text': 'Summarize the previous AI conversations, please.',
        },
    ]
    assert extract_last_message(agent_run) == {
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
    assert flat_messages(messages) == {
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
