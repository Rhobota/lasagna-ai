from .agent_runner import run

from .util import get_name

from .types import (
    AgentRunChained,
    AgentRunExtraction,
    AgentRunMessageList,
    AgentRunParallel,
    AgentSpec,
    AgentRun,
    Cost,
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
    Union, List, Any, Type,
    Callable, Protocol,
)

import pydantic
import json
import copy


def make_model_binder(
    provider: Union[str, ModelFactory],
    model: Union[str, ModelRecord],
    **model_kwargs: Any,
) -> Callable[[AgentCallable], BoundAgentCallable]:
    class ModelBinder():
        def __call__(self, agent: AgentCallable) -> BoundAgentCallable:
            spec: AgentSpec = {
                'agent': agent,
                'provider': provider,
                'model': model,
                'model_kwargs': model_kwargs,
            }
            async def bound_agent(event_callback: EventCallback, prev_runs: List[AgentRun]) -> AgentRun:
                return await run(spec, event_callback, prev_runs)
            for attr in ['__module__', '__qualname__', '__doc__']:
                if hasattr(agent, attr):
                    setattr(bound_agent, attr, getattr(agent, attr))
            bound_agent.__name__ = get_name(agent)
            return bound_agent

        def __str__(self) -> str:
            info = (provider, model, model_kwargs)
            return f'model binder: {info}'

    return ModelBinder()


class PartiallyBoundAgentCallable(Protocol):
    def __call__(
        self,
        **model_kwargs: Any,
    ) -> Callable[[AgentCallable], BoundAgentCallable]: ...


def make_partial_model_binder(
    provider: Union[str, ModelFactory],
    model: Union[str, ModelRecord],
) -> PartiallyBoundAgentCallable:
    class PartialModelBinder():
        def __call__(
            self,
            **model_kwargs: Any,
        ) -> Callable[[AgentCallable], BoundAgentCallable]:
            return make_model_binder(provider, model, **model_kwargs)

        def __str__(self) -> str:
            info = (provider, model)
            return f'partial model binder: {info}'

    return PartialModelBinder()


def _extract_messages_from_tool_result(
    tool_result: ToolResult,
    from_extraction: bool,
) -> List[Message]:
    if tool_result['type'] == 'layered_agent':
        run = tool_result['run']
        return recursive_extract_messages(run, from_tools=True, from_extraction=from_extraction)
    return []


def _recursive_extract_messages_from_tool_res(
    messages: List[Message],
    from_extraction: bool,
) -> List[Message]:
    ms: List[Message] = []
    for m in messages:
        ms.append(m)
        if m['role'] == 'tool_res':
            for t in m['tools']:
                ms.extend(_extract_messages_from_tool_result(t, from_extraction=from_extraction))
    return ms


def recursive_extract_messages(
    agent_run_or_runs: Union[AgentRun, List[AgentRun]],
    from_tools: bool,
    from_extraction: bool,
) -> List[Message]:
    """DFS retrieve all messages within a list of `AgentRuns`."""
    agent_runs: List[AgentRun]
    if isinstance(agent_run_or_runs, list):
        agent_runs = agent_run_or_runs
    else:
        agent_runs = [agent_run_or_runs]
    messages: List[Message] = []
    for run in agent_runs:
        if run['type'] == 'messages':
            messages.extend(
                (
                    _recursive_extract_messages_from_tool_res(run['messages'], from_extraction=from_extraction)
                    if from_tools else
                    run['messages']
                ),
            )
        elif run['type'] == 'chain' or run['type'] == 'parallel':
            messages.extend(
                recursive_extract_messages(run['runs'], from_tools=from_tools, from_extraction=from_extraction),
            )
        elif run['type'] == 'extraction':
            if from_extraction:
                messages.extend(
                    (
                        _recursive_extract_messages_from_tool_res(run['messages'], from_extraction=from_extraction)
                        if from_tools else
                        run['messages']
                    ),
                )
        else:
            raise RuntimeError('unreachable')
    return messages


def extract_last_message(
    agent_run_or_runs: Union[AgentRun, List[AgentRun]],
    from_tools: bool,
    from_extraction: bool,
) -> Message:
    messages = recursive_extract_messages(agent_run_or_runs, from_tools=from_tools, from_extraction=from_extraction)
    if len(messages) == 0:
        raise ValueError('no messages found')
    return messages[-1]


def flat_messages(
    agent_name: str,
    messages: List[Message],
) -> AgentRunMessageList:
    return {
        'agent': agent_name,
        'type': 'messages',
        'messages': messages,
    }


def parallel_runs(
    agent_name: str,
    runs: List[AgentRun],
) -> AgentRunParallel:
    return {
        'agent': agent_name,
        'type': 'parallel',
        'runs': runs,
    }


def chained_runs(
    agent_name: str,
    runs: List[AgentRun],
) -> AgentRunChained:
    return {
        'agent': agent_name,
        'type': 'chain',
        'runs': runs,
    }


def extraction(
    agent_name: str,
    messages: List[Message],
    result: Any,
) -> AgentRunExtraction:
    return {
        'agent': agent_name,
        'type': 'extraction',
        'messages': messages,
        'result': result,
    }


def human_input(prompt: str) -> List[AgentRun]:
    return [
        flat_messages('human_input', [
            {
                'role': 'human',
                'text': prompt,
            },
        ]),
    ]


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


def recursive_sum_costs(
    agent_run_or_runs: Union[AgentRun, List[AgentRun]],
) -> Cost:
    messages = recursive_extract_messages(agent_run_or_runs, from_tools=True, from_extraction=True)

    input_tokens  = [m['cost']['input_tokens']  for m in messages if 'cost' in m and 'input_tokens'  in m['cost']]
    output_tokens = [m['cost']['output_tokens'] for m in messages if 'cost' in m and 'output_tokens' in m['cost']]
    total_tokens  = [m['cost']['total_tokens']  for m in messages if 'cost' in m and 'total_tokens'  in m['cost']]

    cost: Cost = {}

    if input_tokens:
        cost['input_tokens'] = sum(input_tokens)
    if output_tokens:
        cost['output_tokens'] = sum(output_tokens)
    if total_tokens:
        cost['total_tokens'] = sum(total_tokens)

    return cost


async def noop_callback(event: EventPayload) -> None:
    # "noop" mean "no operation" means DON'T DO ANYTHING!
    assert event


MessageExtractor = Callable[[List[AgentRun]], List[Message]]


def build_standard_message_extractor(
    extract_from_tools: bool = False,
    extract_from_extraction: bool = False,
    keep_only_last_human_message: bool = False,
    strip_tool_messages: bool = False,
    system_prompt_override: Union[str, None] = None,
) -> MessageExtractor:
    def extractor(prev_runs: List[AgentRun]) -> List[Message]:
        messages = recursive_extract_messages(
            prev_runs,
            from_tools = extract_from_tools,
            from_extraction = extract_from_extraction,
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


def strip_raw_cost_from_message(message: Message) -> Message:
    message = copy.copy(message)
    if 'raw' in message:
        del message['raw']
    if 'cost' in message:
        del message['cost']
    return message


def strip_raw_cost_from_run(run: AgentRun) -> AgentRun:
    run = copy.copy(run)
    if run['type'] == 'messages' or run['type'] == 'extraction':
        run['messages'] = [strip_raw_cost_from_message(m) for m in run['messages']]
    elif run['type'] == 'chain' or run['type'] == 'parallel':
        run['runs'] = [strip_raw_cost_from_run(r) for r in run['runs']]
    else:
        raise RuntimeError('unreachable')
    return run


def model_dump_all_pydantic_results(run: AgentRun) -> AgentRun:
    run = copy.copy(run)
    if run['type'] == 'messages':
        pass  # noop
    elif run['type'] == 'extraction':
        result = run['result']
        if isinstance(result, pydantic.BaseModel):
            run['result'] = result.model_dump()
    elif run['type'] == 'chain' or run['type'] == 'parallel':
        run['runs'] = [model_dump_all_pydantic_results(r) for r in run['runs']]
    else:
        raise RuntimeError('unreachable')
    return run


def _prep_for_debug_print(run: AgentRun) -> AgentRun:
    run = model_dump_all_pydantic_results(run)
    run = strip_raw_cost_from_run(run)
    return run


def to_str(
    agent_run_or_runs: Union[AgentRun, List[AgentRun]],
) -> str:
    if isinstance(agent_run_or_runs, list):
        runs = agent_run_or_runs
        runs = [_prep_for_debug_print(r) for r in runs]
        return json.dumps(runs, indent=2)
    else:
        run = _prep_for_debug_print(agent_run_or_runs)
        return json.dumps(run, indent=2)
