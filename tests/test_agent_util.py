import pytest

from typing import List

from lasagna.agent_util import (
    bind_model,
    recursive_extract_messages,
    flat_messages,
)

from lasagna.types import (
    Model,
    EventCallback,
    AgentRun,
    Message,
    EventPayload,
)

from lasagna.mock_provider import (
    MockProvider,
)


@bind_model(MockProvider, 'some_model', {'a': 'yes', 'b': 6})
async def my_agent(
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


@pytest.mark.asyncio
async def test_bind_model():
    events = []
    async def event_callback(event: EventPayload) -> None:
        events.append(event)
    prev_runs: List[AgentRun] = []
    new_run = await my_agent(event_callback, prev_runs)
    assert new_run == {
        'agent': 'my_agent',
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
