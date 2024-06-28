import functools

from typing import Union, Dict, Any, List, Callable

from .types import (
    AgentSpec,
    AgentRun,
    ModelFactory,
    AgentCallable,
    BoundAgentCallable,
    EventCallback,
    EventPayload,
    Message,
    Model,
)

from .agent_runner import run


def bind_model(
    provider: Union[str, ModelFactory],
    model: str,
    model_kwargs: Union[Dict[str, Any], None] = None,
) -> Callable[[AgentCallable], BoundAgentCallable]:
    def decorator(agent: AgentCallable) -> BoundAgentCallable:
        spec: AgentSpec = {
            'agent': agent,
            'provider': provider,
            'model': model,
            'model_kwargs': model_kwargs or {},
        }
        @functools.wraps(agent, assigned=['__module__', '__name__', '__qualname__', '__doc__'])
        async def bound_agent(event_callback: EventCallback, prev_runs: List[AgentRun]) -> AgentRun:
            return await run(spec, event_callback, prev_runs)
        return bound_agent
    return decorator


def recursive_extract_messages(agent_runs: List[AgentRun]) -> List[Message]:
    messages: List[Message] = []
    for run in agent_runs:
        if run['type'] == 'messages':
            messages.extend(run['messages'])
        else:
            messages.extend(recursive_extract_messages(run['runs']))
    return messages


def flat_messages(messages: List[Message]) -> AgentRun:
    return {
        'type': 'messages',
        'messages': messages,
    }


def build_most_simple_agent(
    tools: List[Callable] = [],
) -> AgentCallable:
    async def most_simple_agent(
        model: Model,
        event_callback: EventCallback,
        prev_runs: List[AgentRun],
    ) -> AgentRun:
        messages = recursive_extract_messages(prev_runs)
        new_messages = await model.run(event_callback, messages, tools)
        return flat_messages(new_messages)

    return most_simple_agent


async def noop_callback(event: EventPayload) -> None:
    # "noop" mean "no operation" means DON'T DO ANYTHING!
    pass


def extract_last_message(
    agent_run_or_runs: Union[AgentRun, List[AgentRun]],
) -> Message:
    if isinstance(agent_run_or_runs, list):
        messages = recursive_extract_messages(agent_run_or_runs)
    else:
        messages = recursive_extract_messages([agent_run_or_runs])
    if len(messages) == 0:
        raise ValueError('no messages found')
    return messages[-1]
