from .agent_runner import run

from .util import get_name

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
    ToolResult,
)

from typing import (
    Union, Dict, List, Any, Type,
    Callable, Protocol,
)


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
            async def bound_agent(event_callback: EventCallback, prev_runs: List[AgentRun]) -> AgentRun:
                return await run(spec, event_callback, prev_runs)
            for attr in ['__module__', '__qualname__', '__doc__']:
                if hasattr(agent, attr):
                    setattr(bound_agent, attr, getattr(agent, attr))
            bound_agent.__name__ = get_name(agent)
            return bound_agent

        def __str__(self) -> str:
            info = (provider, model, model_kwargs) if model_kwargs else (provider, model)
            return f'model binder: {info}'

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
            return f'partial model binder: {info}'

    return PartialModelBinder()


def _extract_messages_from_tool_result(
    tool_result: ToolResult,
) -> List[Message]:
    if tool_result['type'] == 'layered_agent':
        run = tool_result['run']
        return recursive_extract_messages([run], from_layered_agents=True)
    return []


def _recursive_extract_messages_from_tool_res(
    messages: List[Message],
) -> List[Message]:
    ms: List[Message] = []
    for m in messages:
        ms.append(m)
        if m['role'] == 'tool_res':
            for t in m['tools']:
                ms.extend(_extract_messages_from_tool_result(t))
    return ms


def recursive_extract_messages(
    agent_runs: List[AgentRun],
    from_layered_agents: bool,
) -> List[Message]:
    """DFS retrieve all messages within a list of `AgentRuns`."""
    messages: List[Message] = []
    for run in agent_runs:
        if run['type'] == 'messages':
            messages.extend(
                (
                    _recursive_extract_messages_from_tool_res(run['messages'])
                    if from_layered_agents else
                    run['messages']
                ),
            )
        elif run['type'] == 'chain' or run['type'] == 'parallel':
            messages.extend(
                recursive_extract_messages(run['runs'], from_layered_agents=from_layered_agents),
            )
        elif run['type'] == 'extraction':
            messages.extend(
                (
                    _recursive_extract_messages_from_tool_res(run['messages'])
                    if from_layered_agents else
                    run['messages']
                ),
            )
        else:
            raise RuntimeError('unreachable')
    return messages


def extract_last_message(
    agent_run_or_runs: Union[AgentRun, List[AgentRun]],
    from_layered_agents: bool,
) -> Message:
    if isinstance(agent_run_or_runs, list):
        messages = recursive_extract_messages(agent_run_or_runs, from_layered_agents=from_layered_agents)
    else:
        messages = recursive_extract_messages([agent_run_or_runs], from_layered_agents=from_layered_agents)
    if len(messages) == 0:
        raise ValueError('no messages found')
    return messages[-1]


def flat_messages(
    agent_name: str,
    messages: List[Message],
) -> AgentRun:
    return {
        'agent': agent_name,
        'type': 'messages',
        'messages': messages,
    }


def override_system_prompt(
    messages: List[Message],
    system_prompt: str,
) -> List[Message]:
    sp: Message = {
        'role': 'system',
        'text': system_prompt,
    }
    if messages and messages[0]['role'] == 'system':
        return [sp, *messages[1:]]
    else:
        return [sp, *messages]


def strip_tool_calls_and_results(
    messages: List[Message],
) -> List[Message]:
    return [
        m
        for m in messages
        if m['role'] != 'tool_call' and m['role'] != 'tool_res'
    ]


def strip_all_but_last_human_message(
    messages: List[Message],
) -> List[Message]:
    for m in reversed(messages):
        if m['role'] == 'human':
            return [m]
    return []


async def noop_callback(event: EventPayload) -> None:
    # "noop" mean "no operation" means DON'T DO ANYTHING!
    assert event


MessageExtractor = Callable[[List[AgentRun]], List[Message]]


def build_standard_message_extractor(
    extract_from_layered_agents: bool = False,
    keep_only_last_human_message: bool = False,
    strip_tool_messages: bool = False,
    system_prompt_override: Union[str, None] = None,
) -> MessageExtractor:
    def extractor(prev_runs: List[AgentRun]) -> List[Message]:
        messages = recursive_extract_messages(
            agent_runs = prev_runs,
            from_layered_agents = extract_from_layered_agents,
        )
        if keep_only_last_human_message:
            messages = strip_all_but_last_human_message(messages)
        if strip_tool_messages:
            messages = strip_tool_calls_and_results(messages)
        if system_prompt_override:
            messages = override_system_prompt(messages, system_prompt_override)
        return messages

    return extractor


default_message_extractor = build_standard_message_extractor()


def build_simple_agent(
    name: str,
    tools: List[Callable] = [],
    force_tool: bool = False,
    max_tool_iters: int = 5,
    message_extractor: MessageExtractor = default_message_extractor,
    doc: Union[str, None] = None,
) -> AgentCallable:
    class SimpleAgent():
        async def __call__(
            self,
            model: Model,
            event_callback: EventCallback,
            prev_runs: List[AgentRun],
        ) -> AgentRun:
            messages = message_extractor(prev_runs)
            new_messages = await model.run(
                event_callback = event_callback,
                messages = messages,
                tools = tools,
                force_tool = force_tool,
                max_tool_iters = max_tool_iters,
            )
            return flat_messages(name, new_messages)

        def __str__(self) -> str:
            return name

    a = SimpleAgent()
    if doc:
        a.__doc__ = doc
    return a


def build_extraction_agent(
    name: str,
    extraction_type: Type[ExtractionType],
    message_extractor: MessageExtractor = default_message_extractor,
    doc: Union[str, None] = None,
) -> AgentCallable:
    class ExtractionAgent():
        async def __call__(
            self,
            model: Model,
            event_callback: EventCallback,
            prev_runs: List[AgentRun],
        ) -> AgentRun:
            messages = message_extractor(prev_runs)
            message, result = await model.extract(event_callback, messages, extraction_type)
            return {
                'agent': name,
                'type': 'extraction',
                'messages': [message],
                'result': result,
            }

        def __str__(self) -> str:
            return name

    a = ExtractionAgent()
    if doc:
        a.__doc__ = doc
    return a


def build_agent_chainer(
    name: str,
    agents: List[BoundAgentCallable],
    message_extractor: Union[MessageExtractor, None] = None,
    doc: Union[str, None] = None,
) -> BoundAgentCallable:
    class ChainedAgents():
        async def __call__(
            self,
            event_callback: EventCallback,
            prev_runs: List[AgentRun],
        ) -> AgentRun:
            if message_extractor is not None:
                prev_runs = [
                    flat_messages(
                        agent_name = name,
                        messages = message_extractor(prev_runs),
                    ),
                ]
            else:
                prev_runs = [*prev_runs]  # shallow copy
            new_runs: List[AgentRun] = []
            for agent in agents:
                this_run = await agent(event_callback, prev_runs)
                prev_runs.append(this_run)
                new_runs.append(this_run)
            return {
                'agent': name,
                'type': 'chain',
                'runs': new_runs,
            }

        def __str__(self) -> str:
            return name

    a = ChainedAgents()
    if doc:
        a.__doc__ = doc
    return a


def build_agent_router(
    name: str,
    extraction_type: Type[ExtractionType],
    pick_agent_func: Callable[[ExtractionType], BoundAgentCallable],
    message_extractor: MessageExtractor = default_message_extractor,
    doc: Union[str, None] = None,
) -> AgentCallable:
    class AgentRouter():
        async def __call__(
            self,
            model: Model,
            event_callback: EventCallback,
            prev_runs: List[AgentRun],
        ) -> AgentRun:
            messages = message_extractor(prev_runs)
            message, result = await model.extract(event_callback, messages, extraction_type)
            extraction: AgentRun = {
                'agent': name,
                'type': 'extraction',
                'messages': [message],
                'result': result,
            }
            agent = pick_agent_func(result)
            run = await agent(event_callback, prev_runs)
            return {
                'agent': name,
                'type': 'chain',
                'runs': [
                    extraction,
                    run,
                ],
            }

        def __str__(self) -> str:
            return name

    a = AgentRouter()
    if doc:
        a.__doc__ = doc
    return a


def build_static_output_agent(
    name: str,
    output: str,
    doc: Union[str, None] = None,
) -> AgentCallable:
    class StaticOutputAgent():
        async def __call__(
            self,
            model: Model,
            event_callback: EventCallback,
            prev_runs: List[AgentRun],
        ) -> AgentRun:
            assert model is not None and prev_runs is not None
            await event_callback(('ai', 'text_event', output))
            return flat_messages(name, [
                {
                    'role': 'ai',
                    'text': output,
                },
            ])

        def __str__(self) -> str:
            return name

    a = StaticOutputAgent()
    if doc:
        a.__doc__ = doc
    return a
