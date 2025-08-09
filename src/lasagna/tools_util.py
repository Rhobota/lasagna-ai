import asyncio
import inspect
import json
import collections.abc
from enum import Enum

from typing import (
    List, Dict, Tuple, Union, Any, Callable, Literal,
    get_origin, get_args, cast,
)
from typing_extensions import is_typeddict

from .agent_util import make_model_binder, extract_last_message, flat_messages

from .util import (
    parse_docstring,
    DOCSTRING_PARAM_SUPPORTED_TYPES,
    get_name,
)

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

import logging

_LOG = logging.getLogger(__name__)


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


def _grab_correct_docstring(f: Any) -> str:
    outer_doc = getattr(f, '__doc__', None)
    inner_doc = None
    if not inspect.isfunction(f) and hasattr(f, '__call__'):
        inner_doc = getattr(f.__call__, '__doc__', None)
    return inner_doc or outer_doc or ''


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
        if not type_a_isa_b(expected_param, concrete_param.annotation):
            return False

    if is_async_callable(concrete_callable):
        if get_origin(expected_return) is not collections.abc.Awaitable:
            return False
        eret = get_args(expected_return)[0]
        if not type_a_isa_b(concrete_sig.return_annotation, eret):
            return False

    else:
        if not type_a_isa_b(concrete_sig.return_annotation, expected_return):
            return False

    return True


def type_a_isa_b(a: Any, b: Any) -> bool:
    """
    Correctly compares annotation types such that, e.g., these are equal:
        a = List[dict[str, Set[int]]]
        b = list[Dict[str, set[int]]]

    Also correctly identifies `a isa b`, e.g.
        a = List[str]
        b = Iterable[str]

    Also correctly handles `Union` and `Literal` types.
    """
    a_origin = get_origin(a) or a
    b_origin = get_origin(b) or b
    a_args = get_args(a)
    b_args = get_args(b)

    if b_origin is Any:
        return True

    if a_origin is Union:
        return all(type_a_isa_b(option, b) for option in a_args)
    elif b_origin is Union:
        return any(type_a_isa_b(a, option) for option in b_args)

    if a_origin is Literal and b_origin is Literal:
        return len(set(a_args) - set(b_args)) == 0
    elif a_origin is Literal:
        return all(type_a_isa_b(type(option), b) for option in a_args)
    elif b_origin is Literal:
        _LOG.warning(f'Cannot check against target literal: {a=} {b=}')
        return False

    if is_typeddict(a) and b_origin is dict:
        if not b_args or b_args == (str, Any):
            # Special case we want to handle!
            return True

    try:
        if not (a_origin == b_origin or issubclass(a_origin, b_origin)):
            return False
    except TypeError:
        # Note: Calling `issubclass` on two TypedDicts will land here.
        #       Our equality check above should catch the cases we need, for now.
        #       But, we might need to fix this in the future if we ever need to
        #       compare two TypedDicts that are not the *same* but are *compatible*
        #       with one another.
        return False

    if len(b_args) == 0:
        # Since `b` is totally generic, then `a` must merely be a more
        # specific version, which passes our test.
        return True
    if len(a_args) != len(b_args):
        return False
    for a_arg, b_arg in zip(a_args, b_args):
        if not type_a_isa_b(a_arg, b_arg):
            return False

    return True


def get_tool_params(tool: Callable) -> Tuple[str, List[ToolParam]]:
    doc_params: List[ToolParam]
    description, doc_params = parse_docstring(_grab_correct_docstring(tool))

    is_agent_callable = is_callable_of_type(tool, AgentCallable, no_throw=True)
    is_bound_agent_callable = is_callable_of_type(tool, BoundAgentCallable, no_throw=True)

    if is_agent_callable or is_bound_agent_callable:
        pass  # no validation on these

    else:
        concrete_sig = inspect.signature(tool)
        concrete_params = concrete_sig.parameters.values()

        if len(concrete_params) != len(doc_params):
            raise ValueError(f'tool `{get_name(tool)}` has parameter length mismatch: tool has {len(concrete_params)}, docstring has {len(doc_params)}')

        for concrete_param, doc_param in zip(concrete_params, doc_params):
            if concrete_param.name != doc_param['name']:
                raise ValueError(f"tool `{get_name(tool)}` has parameter name mismatch: tool name is `{concrete_param.name}`, docstring name is `{doc_param['name']}`")
            if doc_param.get('optional', False):
                if concrete_param.default is inspect.Parameter.empty:
                    raise ValueError(f"tool `{get_name(tool)}` has an optional parameter without a default value: `{concrete_param.name}`")
            if concrete_param.annotation is inspect.Parameter.empty:
                _LOG.warning(f'tool `{get_name(tool)}` is missing annotation for parameter `{concrete_param.name}`')
            else:
                if doc_param['type'].startswith('enum '):
                    if issubclass(concrete_param.annotation, str):
                        pass  # a str can accept any value, so we're good
                    elif issubclass(concrete_param.annotation, Enum):
                        concrete_enum_vals = set([
                            e.value
                            for e in concrete_param.annotation.__members__.values()
                        ])
                        doc_enum_vals = set(doc_param['type'].split()[1:])
                        if concrete_enum_vals != doc_enum_vals:
                            raise ValueError(f"tool `{get_name(tool)}` has parameter `{concrete_param.name}` enum value mismatch: tool has enum values `{sorted(concrete_enum_vals)}`, docstring has enum values `{sorted(doc_enum_vals)}`")
                    else:
                        raise ValueError(f"tool `{get_name(tool)}` has parameter `{concrete_param.name}` type mismatch: tool type is `{concrete_param.annotation}`, docstring type is `{doc_param['type']}`")
                else:
                    doc_param_type = DOCSTRING_PARAM_SUPPORTED_TYPES.get(doc_param['type'], 'unknown')
                    if not type_a_isa_b(doc_param_type, concrete_param.annotation):
                        raise ValueError(f"tool `{get_name(tool)}` has parameter `{concrete_param.name}` type mismatch: tool type is `{concrete_param.annotation}`, docstring type is `{doc_param_type}`")

    return description, doc_params


def validate_args(tool: Callable, parsed_args: Dict) -> Dict:
    tool_name = get_name(tool)
    _, doc_params = parse_docstring(_grab_correct_docstring(tool))
    concrete_sig = inspect.signature(tool)
    concrete_params = concrete_sig.parameters.values()

    # Check for unexpected parameters:
    unexpected_params: List[str] = []
    doc_param_names = set([doc_param['name'] for doc_param in doc_params])
    for arg in sorted(parsed_args.keys()):
        if arg not in doc_param_names:
            unexpected_params.append(arg)
    if unexpected_params:
        n = len(unexpected_params)
        s = ', '.join([repr(m) for m in unexpected_params])
        if n == 1:
            raise TypeError(f"{tool_name}() got 1 unexpected argument: {s}")
        else:
            raise TypeError(f"{tool_name}() got {n} unexpected arguments: {s}")

    # Check for missing parameters:
    missing_params: List[str] = []
    for doc_param in doc_params:
        name = doc_param['name']
        optional = doc_param.get('optional', False)
        if not optional and name not in parsed_args:
            missing_params.append(name)
    if missing_params:
        n = len(missing_params)
        s = ', '.join([repr(m) for m in missing_params])
        if n == 1:
            raise TypeError(f"{tool_name}() missing 1 required argument: {s}")
        else:
            raise TypeError(f"{tool_name}() missing {n} required arguments: {s}")

    # Check types:
    validated_args: Dict[str, Any] = {}
    doc_param_type_lookup = {
        doc_param['name']: doc_param['type']
        for doc_param in doc_params
    }
    concrete_annotation_lookup = {
        concrete_param.name: concrete_param.annotation
        for concrete_param in concrete_params
    }
    for arg, value in parsed_args.items():
        assert arg in doc_param_type_lookup
        type_str = doc_param_type_lookup[arg]
        if type_str.startswith('enum '):
            value_str = str(value)
            doc_enum_vals = set(type_str.split()[1:])
            if value_str not in doc_enum_vals:
                raise TypeError(f"{tool_name}() got invalid value for argument `{arg}`: {repr(value)} (valid values are {sorted(doc_enum_vals)})")
            concrete_annotation = concrete_annotation_lookup[arg]
            if concrete_annotation is inspect.Parameter.empty:
                validated_args[arg] = value_str
            elif issubclass(concrete_annotation, str):
                validated_args[arg] = value_str
            else:
                assert issubclass(concrete_annotation, Enum)  # we know this from the tool passing `get_tool_params()`
                concrete_enum_lookup = {
                    e.value: e
                    for e in concrete_annotation.__members__.values()
                }
                assert value_str in concrete_enum_lookup  # we know this from the tool passing `get_tool_params()`
                validated_args[arg] = concrete_enum_lookup[value_str]
        else:
            type_ = DOCSTRING_PARAM_SUPPORTED_TYPES[type_str]
            try:
                validated_args[arg] = type_(value)  # <-- assume c'tor will throw as necessary
            except Exception as e:
                raise TypeError(f"{tool_name}() got invalid value for argument `{arg}`: {repr(value)} ({e})")
    return validated_args


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

        if not isinstance(parsed_args, dict):
            raise TypeError("tool output must be a JSON object")

        parsed_args = validate_args(func, parsed_args)

        is_agent_callable = is_callable_of_type(func, AgentCallable, no_throw=True)
        is_bound_agent_callable = is_callable_of_type(func, BoundAgentCallable, no_throw=True)

        if is_agent_callable or is_bound_agent_callable:
            messages: List[Message] = [*prev_messages]  # shallow copy
            if len(parsed_args) > 0:
                messages.append({
                    'role': 'tool_call',
                    'tools': [tool_call],
                })
            prev_runs: List[AgentRun] = [
                flat_messages(
                    agent_name = 'upstream_agent',
                    messages = messages,
                ),
            ]

            if is_agent_callable:
                assert model_spec is not None
                agent = cast(AgentCallable, func)
                binder = make_model_binder(
                    provider = model_spec['provider'],
                    model = model_spec['model'],
                    **model_spec.get('model_kwargs', {}),
                )
                bound_agent = binder(agent)
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
        _LOG.exception(e)
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
        last_message = extract_last_message([tool_result['run']], from_tools=True, from_extraction=True)
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
