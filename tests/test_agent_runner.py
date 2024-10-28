import pytest

from lasagna.agent_runner import run

from lasagna.types import (
    AgentSpec,
    AgentRun,
    EventCallback,
    Message,
    Model,
    EventPayload,
)

from lasagna.registrar import (
    register_agent,
    register_provider,
    AGENTS,
    PROVIDERS,
)

from lasagna.mock_provider import MockProvider

from typing import List


async def agent_1(
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
async def test_run_with_registered_names():
    AGENTS.clear()
    PROVIDERS.clear()
    register_agent('agent_1', 'Agent 1', agent_1)
    register_provider('mock_provider', 'Mock Provider', MockProvider, [])
    spec: AgentSpec = {
        'agent': 'agent_1',
        'provider': 'mock_provider',
        'model': 'some_model',
        'model_kwargs': {
            'b': 6,
            'a': 'yes',
        },
    }
    events: List[EventPayload] = []
    async def event_callback(event: EventPayload) -> None:
        events.append(event)
    prev_runs: List[AgentRun] = []
    new_run = await run(spec, event_callback, prev_runs)
    assert new_run == {
        'agent': 'agent_1',
        'provider': 'mock_provider',
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
        ('agent', 'start', 'agent_1'),
        ('ai', 'text_event', 'Hi!'),
        ('agent', 'end', new_run),
    ]


@pytest.mark.asyncio
async def test_run_direct():
    AGENTS.clear()
    PROVIDERS.clear()
    spec: AgentSpec = {
        'agent': agent_1,
        'provider': MockProvider,
        'model': 'some_model',
        'model_kwargs': {
            'b': 6,
            'a': 'yes',
        },
    }
    events: List[EventPayload] = []
    async def event_callback(event: EventPayload) -> None:
        events.append(event)
    prev_runs: List[AgentRun] = []
    new_run = await run(spec, event_callback, prev_runs)
    assert new_run == {
        'agent': 'agent_1',
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
        ('agent', 'start', 'agent_1'),
        ('ai', 'text_event', 'Hi!'),
        ('agent', 'end', new_run),
    ]
