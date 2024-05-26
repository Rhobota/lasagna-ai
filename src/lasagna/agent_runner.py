from .registrar import AGENTS, MODEL_PROVIDERS

from .types import (
    AgentSpec,
    AgentCallable,
    ProviderFactory,
    ChatMessage,
    EventCallback,
)

from typing import List


async def run(
    agent_spec: AgentSpec,
    event_callback: EventCallback,
    messages: List[ChatMessage],
) -> List[ChatMessage]:
    agent: AgentCallable
    if isinstance(agent_spec['agent'], str):
        agent = AGENTS[agent_spec['agent']]['runner']
    else:
        agent = agent_spec['agent']

    provider: ProviderFactory
    if isinstance(agent_spec['provider'], str):
        provider = MODEL_PROVIDERS[agent_spec['provider']]['factory']
    else:
        provider = agent_spec['provider']

    kwargs = agent_spec.get('model_kwargs', None)
    if kwargs is None:
        kwargs = {}

    llm = provider(model=agent_spec['model'], **kwargs)

    return await agent(
        llm,
        event_callback,
        messages,
    )
