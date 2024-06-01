from .registrar import AGENTS, PROVIDERS

from .types import (
    AgentSpec,
    AgentCallable,
    ModelFactory,
    ChatMessage,
    EventCallback,
)

from .known_providers import attempt_load_known_providers

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

    model_factory: ModelFactory
    if isinstance(agent_spec['provider'], str):
        provider_str = agent_spec['provider']
        if provider_str not in PROVIDERS:
            attempt_load_known_providers(provider_str)
        model_factory = PROVIDERS[provider_str]['factory']
    else:
        model_factory = agent_spec['provider']

    kwargs = agent_spec.get('model_kwargs', None)
    if kwargs is None:
        kwargs = {}

    model = model_factory(model=agent_spec['model'], **kwargs)

    return await agent(
        model,
        event_callback,
        messages,
    )
