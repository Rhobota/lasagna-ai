from .registrar import AGENTS, PROVIDERS

from .util import get_name

from .types import (
    AgentSpec,
    AgentRun,
    AgentCallable,
    ModelFactory,
    EventCallback,
)

from .known_providers import attempt_load_known_providers

from typing import List


async def run(
    agent_spec: AgentSpec,
    event_callback: EventCallback,
    prev_runs: List[AgentRun],
) -> AgentRun:
    agent: AgentCallable
    agent_name: str
    if isinstance(agent_spec['agent'], str):
        agent_name = agent_spec['agent']
        agent = AGENTS[agent_name]['runner']
    else:
        agent = agent_spec['agent']
        agent_name = get_name(agent)

    model_name = agent_spec['model'] \
        if isinstance(agent_spec['model'], str) \
        else agent_spec['model']['formal_name']

    model_kwargs = agent_spec.get('model_kwargs', None)
    if model_kwargs is None:
        model_kwargs = {}

    model_factory: ModelFactory
    provider_str: str
    if isinstance(agent_spec['provider'], str):
        provider_str = agent_spec['provider']
        if provider_str not in PROVIDERS:
            attempt_load_known_providers(provider_str)
        model_factory = PROVIDERS[provider_str]['factory']
        model = model_factory(model=model_name, **model_kwargs)
    else:
        model_factory = agent_spec['provider']
        model = model_factory(model=model_name, **model_kwargs)
        provider_str = get_name(model.__class__)

    await event_callback(('agent', 'start', agent_name))

    agent_run = await agent(
        model,
        event_callback,
        prev_runs,
    )

    if 'agent' not in agent_run:
        agent_run['agent'] = agent_name
    if 'provider' not in agent_run:
        agent_run['provider'] = provider_str
    if 'model' not in agent_run:
        agent_run['model'] = model_name
    if 'model_kwargs' not in agent_run and 'model_kwargs' in agent_spec:
        agent_run['model_kwargs'] = model_kwargs

    await event_callback(('agent', 'end', agent_run))

    return agent_run
