import functools

from typing import Union, Dict, Any, List, Callable

from .types import (
    AgentSpec,
    AgentRun,
    ModelFactory,
    AgentCallable,
    BoundAgentCallable,
    EventCallback,
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
        async def bound_agent(event_callback: EventCallback, messages: List[AgentRun]) -> AgentRun:
            return await run(spec, event_callback, messages)
        return bound_agent
    return decorator
