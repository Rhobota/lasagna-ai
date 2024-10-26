import asyncio
import inspect
import json
import collections.abc

from typing import (
    List, Dict, Tuple, Union, Any, Callable,
    get_origin, get_args,
)

from .util import parse_docstring

from .types import (
    AgentCallable,
    BoundAgentCallable,
    Message,
    ToolCall,
    ToolResult,
    ToolParam,
)


LAYERED_AGENT_TYPES = [AgentCallable, BoundAgentCallable]


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


def get_name(obj: Any) -> str:
    name = str(obj.__name__) if hasattr(obj, '__name__') else str(obj)
    return name


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
    if any([is_callable_of_type(tool, t, no_throw=True) for t in LAYERED_AGENT_TYPES]):
        if len(params) == 0:
            params = [
                {
                    'name': 'prompt',
                    'type': 'str',
                },
            ]
        elif len(params) == 1:
            p = params[0]
            if p['name'] != 'prompt' or p['type'] != 'str' or p.get('optional', False):
                raise ValueError('Layered agents must take a single string parameter named `prompt`.')
        else:
            raise ValueError('Layered agents must take a single string parameter named `prompt`.')
    return description, params


async def _run_single_tool(
    tool_call: ToolCall,
    tools_map: Dict[str, Callable],
) -> ToolResult:
    call_id = 'unknown'

    try:
        call_id = tool_call['call_id']
        func = tools_map[tool_call['function']['name']]
        args = tool_call['function']['arguments']

        if any([is_callable_of_type(func, t, no_throw=True) for t in LAYERED_AGENT_TYPES]):
            raise RuntimeError('not supported yet')

        else:
            if is_async_callable(func):
                res = await func(**json.loads(args))
            else:
                def _wrapped_sync() -> Any:
                    return func(**json.loads(args))
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
    messages: List[Message],
    tools_map: Dict[str, Callable],
) -> Union[List[ToolResult], None]:
    assert len(messages) > 0
    tool_messages = [message for message in messages if message['role'] == 'tool_call']
    if len(tool_messages) == 0:
        return None

    to_gather: List[asyncio.Task[ToolResult]] = []

    for message in tool_messages:
        for t in message['tools']:
            assert t['call_type'] == 'function'
            to_gather.append(asyncio.create_task(
                _run_single_tool(t, tools_map),
            ))

    return await asyncio.gather(*to_gather)


def build_tool_response_message(tool_results: List[ToolResult]) -> Message:
    return {
        'role': 'tool_res',
        'tools': tool_results,
    }
