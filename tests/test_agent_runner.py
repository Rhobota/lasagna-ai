import pytest

from lasagna.agent_runner import run

from lasagna.types import (
    AgentSpec,
    EventCallback,
    ChatMessage,
    LLM,
    EventPayload,
)

from lasagna.registrar import (
    register_agent,
    register_model_provider,
    AGENTS,
    MODEL_PROVIDERS,
)

from typing import List, Dict, Any, Callable


class MockProvider(LLM):
    def __init__(self, model: str, **model_kwargs: Dict[str, Any]):
        self.model = model
        self.model_kwargs = model_kwargs

    async def run(
        self,
        event_callback: EventCallback,
        messages: List[ChatMessage],
        tools: List[Callable],
        force_tool: bool = False,
        max_tool_iters: int = 5,
    ) -> List[ChatMessage]:
        event: EventPayload = 'ai', 'text_event', 'Hi!'
        await event_callback(event)
        res: List[ChatMessage] = [
            {
                'role': 'ai',
                'text': f"model: {self.model}",
                'cost': None,
                'raw': None,
            },
        ]
        for key in sorted(self.model_kwargs.keys()):
            val = self.model_kwargs[key]
            m: ChatMessage = {
                'role': 'human',
                'text': f"model_kwarg: {key} = {val}",
                'cost': None,
                'raw': None,
            }
            res.append(m)
        return res


async def agent_1(
    llm: LLM,
    event_callback: EventCallback,
    messages: List[ChatMessage],
) -> List[ChatMessage]:
    new_messages = await llm.run(event_callback, messages, [])
    return new_messages


@pytest.mark.asyncio
async def test_run_with_registered_names():
    AGENTS.clear()
    MODEL_PROVIDERS.clear()
    register_agent('agent_1', 'Agent 1', agent_1)
    register_model_provider('mock_provider', 'Mock Provider', MockProvider, [])
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
    messages: List[ChatMessage] = []
    new_messages = await run(spec, event_callback, messages)
    assert new_messages == [
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
    ]
    assert events == [
        ('ai', 'text_event', 'Hi!'),
    ]


@pytest.mark.asyncio
async def test_run_direct():
    AGENTS.clear()
    MODEL_PROVIDERS.clear()
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
    messages: List[ChatMessage] = []
    new_messages = await run(spec, event_callback, messages)
    assert new_messages == [
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
    ]
    assert events == [
        ('ai', 'text_event', 'Hi!'),
    ]
