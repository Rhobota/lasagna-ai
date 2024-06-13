import pytest

from typing import List

from lasagna.agent_util import (
    bind_model,
)

from lasagna.types import (
    Model,
    EventCallback,
    AgentRun,
    Message,
    EventPayload,
)

from .test_agent_runner import (
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
