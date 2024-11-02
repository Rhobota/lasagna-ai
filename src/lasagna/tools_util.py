import asyncio
import inspect
import json
import collections.abc

from typing import (
    List, Dict, Tuple, Union, Any, Callable,
    get_origin, get_args, cast,
)

from .agent_util import bind_model, extract_last_message, flat_messages

from .util import parse_docstring, get_name

from .types import (
    AgentCallable,
    AgentRun,
    BoundAgentCallable,
    EventCallback,
    Message,
    ModelSpec,
    ToolCall,
    ToolResult,
    ToolParam,
)


def convert_to_json_schema(params: List[ToolParam]) -> Dict[str, object]:
    def convert_type(t: str) -> Dict[str, object]:
        if t.startswith('enum '):
            return {
                "type": "string",
                "enum": t.split()[1:],
            }
        else:
            return {
                "type": {
                    'str': 'string',
                    'float': 'number',
                    'int': 'integer',
                    'bool': 'boolean',
                }[t],
            }
    return {
        "type": "object",
        "properties": {
            p['name']: {
                **convert_type(p['type']),
                **({
                    "description": p['description'],
                } if 'description' in p and p['description'] else {}),
            }
            for p in params
        },
        "required": [
            p['name']
            for p in params
            if not p.get('optional', False)
        ],
        "additionalProperties": False,
    }


def is_async_callable(f: Any) -> bool:
    if not inspect.isfunction(f) and hasattr(f, '__call__'):
        f = f.__call__
    return callable(f) and inspect.iscoroutinefunction(f)


def is_callable_of_type(
    concrete_callable: Any,
    expected_callable_type: Any,
    no_throw: bool = False,
) -> bool:
    if get_origin(expected_callable_type) is not collections.abc.Callable:
        if no_throw:
            return False
        raise ValueError("`expected_callable_type` must be a Callable type")

    expected_params, expected_return = get_args(expected_callable_type)

    if not callable(concrete_callable):
        if no_throw:
            return False
        raise ValueError("`concrete_callable` must be a callable object")

    concrete_sig = inspect.signature(concrete_callable)
    concrete_params = concrete_sig.parameters.values()

    for concrete_param in concrete_params:
        if concrete_param.annotation is inspect.Parameter.empty:
            if no_throw:
                return False
            raise ValueError(f"`concrete_callable` must have fully-specified types: missing type for `{concrete_param.name}`")

    if concrete_sig.return_annotation is inspect.Parameter.empty:
        if no_throw:
            return False
        raise ValueError(f"`concrete_callable` must have fully-specified types: missing return type")

    if len(concrete_params) != len(expected_params):
        return False

    for concrete_param, expected_param in zip(concrete_params, expected_params):
        if concrete_param.annotation != expected_param:
            return False

    if is_async_callable(concrete_callable):
        if get_origin(expected_return) is not collections.abc.Awaitable:
            return False
        eret = get_args(expected_return)[0]
        if concrete_sig.return_annotation != eret:
            return False

    else:
        if concrete_sig.return_annotation != expected_return:
            return False

    return True


def get_tool_params(tool: Callable) -> Tuple[str, List[ToolParam]]:
    params: List[ToolParam]
    description, params = parse_docstring(tool.__doc__ or '')
    return description, params


async def _run_single_tool(
    tool_call: ToolCall,
    prev_messages: List[Message],
    tools_map: Dict[str, Callable],
    event_callback: EventCallback,
    model_spec: Union[ModelSpec, None],
) -> ToolResult:
    call_id = 'unknown'

    try:
        call_id = tool_call['call_id']
        func = tools_map[tool_call['function']['name']]
        args = tool_call['function']['arguments']

        parsed_args = json.loads(args)

        is_agent_callable = is_callable_of_type(func, AgentCallable, no_throw=True)
        is_bound_agent_callable = is_callable_of_type(func, BoundAgentCallable, no_throw=True)

        if model_spec and (is_agent_callable or is_bound_agent_callable):
            messages: List[Message] = [*prev_messages]  # shallow copy
            if len(parsed_args) > 0:
                messages.append({
                    'role': 'tool_call',
                    'tools': [tool_call],
                })
            prev_runs: List[AgentRun] = [flat_messages(messages)]

            if is_agent_callable:
                agent = cast(AgentCallable, func)
                bound_agent = bind_model(**model_spec)(agent)
            elif is_bound_agent_callable:
                bound_agent = cast(BoundAgentCallable, func)
            else:
                raise RuntimeError('unreachable')

            this_run = await bound_agent(event_callback, prev_runs)

            return {
                'type': 'layered_agent',
                'call_id': call_id,
                'run': this_run,
            }

        else:
            if is_async_callable(func):
                res = await func(**parsed_args)
            else:
                def _wrapped_sync() -> Any:
                    return func(**parsed_args)
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(None, _wrapped_sync)
            return {
                'type': 'any',
                'call_id': call_id,
                'result': res,
            }

    except Exception as e:
        error = f"{get_name(type(e))}: {e}"
        return {
            'type': 'any',
            'call_id': call_id,
            'result': error,
            'is_error': True,
        }


async def handle_tools(
    prev_messages: List[Message],
    new_messages: List[Message],
    tools_map: Dict[str, Callable],
    event_callback: EventCallback,
    model_spec: Union[ModelSpec, None],
) -> Union[List[ToolResult], None]:
    tool_messages = [message for message in new_messages if message['role'] == 'tool_call']
    if len(tool_messages) == 0:
        return None

    to_gather: List[asyncio.Task[ToolResult]] = []

    for message in tool_messages:
        for t in message['tools']:
            assert t['call_type'] == 'function'
            to_gather.append(asyncio.create_task(
                _run_single_tool(
                    t,
                    prev_messages,
                    tools_map,
                    event_callback,
                    model_spec,
                ),
            ))

    return await asyncio.gather(*to_gather)


def extract_tool_result_as_sting(tool_result: ToolResult) -> str:
    content: str

    if tool_result['type'] == 'any':
        content = str(tool_result['result'])

    elif tool_result['type'] == 'layered_agent':
        last_message = extract_last_message([tool_result['run']], from_layered_agents=True)
        if last_message['role'] == 'ai':
            content = last_message['text'] or ''
        elif last_message['role'] == 'tool_call':
            content = json.dumps([
                {
                    'name': c['function']['name'],
                    'values': json.loads(c['function']['arguments']),
                }
                for c in last_message['tools']
            ])
        elif last_message['role'] == 'tool_res':
            content = '\n'.join([
                extract_tool_result_as_sting(r)
                for r in last_message['tools']
            ])
        else:
            raise RuntimeError('unreachable')

    else:
        raise RuntimeError('unreachable')

    return content


def build_tool_response_message(tool_results: List[ToolResult]) -> Message:
    return {
        'role': 'tool_res',
        'tools': tool_results,
    }
