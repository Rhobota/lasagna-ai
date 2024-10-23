import pytest

from typing import Callable, List

from lasagna.agent_util import (
    bind_model,
    noop_callback,
    partial_bind_model,
    recursive_extract_messages,
    flat_messages,
    build_extraction_agent,
)

from lasagna.types import (
    AgentCallable,
    BoundAgentCallable,
    Model,
    EventCallback,
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


async def my_run_agent(
    model: Model,
    event_callback: EventCallback,
    prev_runs: List[AgentRun],
) -> AgentRun:
    assert len(prev_runs) == 0
    messages: List[Message] = []
    new_messages = await model.run(event_callback, messages, [])
    return {
        'type': 'messages',
        'messages': new_messages,
    }


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
        'agent': 'my_run_agent',
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
        ('ai', 'text_event', 'Hi!'),
    ]


@pytest.mark.asyncio
async def test_bind_model():
    my_binder = bind_model(MockProvider, 'some_model', {'a': 'yes', 'b': 6})
    await _agent_common_test(my_binder, my_run_agent)


@pytest.mark.asyncio
async def test_partial_bind_model():
    my_binder = partial_bind_model(MockProvider, 'some_model')({'a': 'yes', 'b': 6})
    await _agent_common_test(my_binder, my_run_agent)


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
    ]
    assert run['type'] == 'messages'
    ms = run['messages']
    assert len(ms) == 1
    m = ms[0]
    assert m['role'] == 'extraction'
    p = m['parsed']
    assert isinstance(p, MyTestType)
    assert p.a == 'yes'
    assert p.b == 6


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
