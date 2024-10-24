import functools

from typing import Union, Dict, Any, List, Callable, Protocol, Type

from .types import (
    AgentSpec,
    AgentRun,
    ModelFactory,
    ModelRecord,
    AgentCallable,
    BoundAgentCallable,
    EventCallback,
    EventPayload,
    Message,
    Model,
    ExtractionType,
)

from .agent_runner import run


def bind_model(
    provider: Union[str, ModelFactory],
    model: Union[str, ModelRecord],
    model_kwargs: Union[Dict[str, Any], None] = None,
) -> Callable[[AgentCallable], BoundAgentCallable]:
    class ModelBinder():
        def __call__(self, agent: AgentCallable) -> BoundAgentCallable:
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

        def __str__(self) -> str:
            info = (provider, model, model_kwargs) if model_kwargs else (provider, model)
            return str(info)

    return ModelBinder()


class PartiallyBoundAgentCallable(Protocol):
    def __call__(
        self,
        model_kwargs: Union[Dict[str, Any], None] = None,
    ) -> Callable[[AgentCallable], BoundAgentCallable]: ...


def partial_bind_model(
    provider: Union[str, ModelFactory],
    model: Union[str, ModelRecord],
) -> PartiallyBoundAgentCallable:
    class PartialModelBinder():
        def __call__(
            self,
            model_kwargs: Union[Dict[str, Any], None] = None,
        ) -> Callable[[AgentCallable], BoundAgentCallable]:
            return bind_model(provider, model, model_kwargs)

        def __str__(self) -> str:
            info = (provider, model)
            return str(info)

    return PartialModelBinder()


def recursive_extract_messages(agent_runs: List[AgentRun]) -> List[Message]:
    messages: List[Message] = []
    for run in agent_runs:
        if run['type'] == 'messages':
            messages.extend(run['messages'])
        elif run['type'] == 'chain' or run['type'] == 'parallel':
            messages.extend(recursive_extract_messages(run['runs']))
        elif run['type'] == 'extraction':
            messages.append(run['message'])
        else:
            raise RuntimeError('unreachable')
    return messages


def flat_messages(messages: List[Message]) -> AgentRun:
    return {
        'type': 'messages',
        'messages': messages,
    }


def build_most_simple_agent(
    tools: List[Callable] = [],
) -> AgentCallable:
    class MostSimpleAgent():
        async def __call__(
            self,
            model: Model,
            event_callback: EventCallback,
            prev_runs: List[AgentRun],
        ) -> AgentRun:
            messages = recursive_extract_messages(prev_runs)
            new_messages = await model.run(event_callback, messages, tools)
            return flat_messages(new_messages)

        def __str__(self) -> str:
            if tools:
                tool_names = ', '.join([tool.__name__ for tool in tools])
                return f'simple agent with tools: {tool_names}'
            else:
                return 'simple agent *no* tools'

    return MostSimpleAgent()


def build_extraction_agent(
    extraction_type: Type[ExtractionType],
) -> AgentCallable:
    class ExtractionAgent():
        async def __call__(
            self,
            model: Model,
            event_callback: EventCallback,
            prev_runs: List[AgentRun],
        ) -> AgentRun:
            messages = recursive_extract_messages(prev_runs)
            message, result = await model.extract(event_callback, messages, extraction_type)
            return {
                'type': 'extraction',
                'message': message,
                'result': result,
            }

        def __str__(self) -> str:
            return f'extraction agent with type: {extraction_type}'

    return ExtractionAgent()


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
