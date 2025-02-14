"""
This module is the Lasagna adapter for the Ollama server.

For more information about Ollama, see:
 - https://ollama.com/
"""

from .types import (
    Cost,
    EventPayload,
    MessageToolCall,
    ModelSpec,
    Message,
    Media,
    EventCallback,
    ToolCall,
    Model,
    ExtractionType,
)

from .util import (
    convert_to_image_base64,
    exponential_backoff_retry_delays,
    get_name,
    recursive_hash,
)

from .tools_util import (
    convert_to_json_schema,
    extract_tool_result_as_sting,
    get_tool_params,
    handle_tools,
    build_tool_response_message,
)

from .pydantic_util import ensure_pydantic_model, build_and_validate

from openai.lib._pydantic import to_strict_json_schema

from typing import (
    List, Callable, AsyncIterator, Any, Type,
    Tuple, Dict, Union,
    cast,
)

import os
import asyncio
import httpx
import copy
import json

import logging

_LOG = logging.getLogger(__name__)


def _convert_to_ollama_tool(tool: Callable) -> Dict:
    description, params = get_tool_params(tool)
    return {
        'type': 'function',
        'function': {
            'name': get_name(tool),
            'description': description,
            'parameters': convert_to_json_schema(params),
        },
    }


def _convert_to_ollama_tools(tools: List[Callable]) -> Union[None, List[Dict]]:
    if len(tools) == 0:
        return None
    specs = [_convert_to_ollama_tool(tool) for tool in tools]
    return specs


def _log_dumps(val: Any) -> str:
    if isinstance(val, dict):
        return json.dumps(val)
    else:
        return str(val)


async def _convert_to_ollama_media(media: List[Media]) -> Dict:
    res: Dict = {
        'images': [],
    }
    for m in media:
        assert m['type'] == 'image'
        mimetype, data = await convert_to_image_base64(m['image'])
        assert mimetype
        res['images'].append(data)
    return res


def _convert_to_ollama_tool_calls(tools: List[ToolCall]) -> List:
    res: List = []
    for t in tools:
        assert t['call_type'] == 'function'
        res.append({
            # t['call_id'] NOT USED! Ollama doesn't do that sort of thing.
            'function': {
                'name': t['function']['name'],
                'arguments': json.loads(t['function']['arguments']),
            },
        })
    return res


async def _convert_to_ollama_messages(messages: List[Message]) -> List[Dict]:
    res: List[Dict] = []
    map = {
        'system': 'system',
        'human': 'user',
        'ai': 'assistant',
    }
    prev_tool_call_map = {}
    for m in messages:
        if m['role'] == 'system' or m['role'] == 'human' or m['role'] == 'ai':  # <-- not using boolean 'in' to make mypy happy
            media = {}
            if 'media' in m and m['media']:
                media = await _convert_to_ollama_media(m['media'])
            res.append({
                'role': map[m['role']],
                'content': m.get('text', ''),
                **media,
            })
        elif m['role'] == 'tool_call':
            res.append({
                'role': 'assistant',
                'content': '',
                'tool_calls': _convert_to_ollama_tool_calls(m['tools']),
            })
            prev_tool_call_map = {
                t['call_id']: t['function']['name']
                for t in m['tools']
            }
        elif m['role'] == 'tool_res':
            for t in m['tools']:
                # t['is_error'] IS NOT USED!
                res.append({
                    'role': 'tool',
                    'content': extract_tool_result_as_sting(t),
                    'name': prev_tool_call_map.get(t['call_id'], 'unknown'),
                })
        else:
            raise RuntimeError(f"unreachable: {m['role']}")
    return res


async def _event_stream(
    url: str,
    payload: Dict,
    timeout_seconds: float,
) -> AsyncIterator[Dict]:
    timeout = httpx.Timeout(timeout_seconds, connect=2.0)  # 2s timeout on connect. `timeout_seconds` elsewhere (read/write/pool).
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream('POST', url, json=payload) as r:
            if r.status_code != 200:
                error_text = json.loads(await r.aread())['error']
                raise RuntimeError(f'Ollama error: {error_text}')
            async for line in r.aiter_lines():
                rec = json.loads(line)
                if rec.get('error'):
                    error_text = rec['error']
                    raise RuntimeError(f'Ollama error: {error_text}')
                yield rec


def _set_cost_raw(message: Message, raw: List[Dict]) -> Message:
    cost: Cost = {
        'input_tokens': None,
        'output_tokens': None,
        'total_tokens': None,
    }
    for event in raw:
        if 'prompt_eval_count' in event:
            cost['input_tokens'] = (cost['input_tokens'] or 0) + event['prompt_eval_count']
        if 'eval_count' in event:
            cost['output_tokens'] = (cost['output_tokens'] or 0) + event['eval_count']
    cost['total_tokens'] = (cost['input_tokens'] or 0) + (cost['output_tokens'] or 0)
    new_message: Message = copy.copy(message)
    if (cost['total_tokens'] or 0) > 0:
        new_message['cost'] = cost
    new_message['raw'] = raw
    return new_message


async def _process_stream(
    stream: AsyncIterator[Dict],
    event_callback: EventCallback,
) -> List[Message]:
    raw: List[Dict] = []
    content: List[str] = []
    tools: List[ToolCall] = []
    async for event in stream:
        raw.append(event)
        if 'message' in event:
            m = event['message']
            if 'content' in m:
                c = m['content']
                if c:
                    assert isinstance(c, str)
                    await event_callback(('ai', 'text_event', c))
                    content.append(c)
            if 'tool_calls' in m:
                for i, t in enumerate(m['tool_calls']):
                    assert 'function' in t
                    f = t['function']
                    name = f['name']
                    args = f['arguments']  # is a dict
                    args_str = json.dumps(args)
                    tool_call: ToolCall = {
                        'call_id': f'call_{i}',  # <-- Ollama doesn't do the `call_id` thing, so we invent a call_id
                        'call_type': 'function',
                        'function': {
                            'name': name,
                            'arguments': args_str,
                        },
                    }
                    await event_callback(('tool_call', 'text_event', f'{name}({args_str})\n'))
                    await event_callback(('tool_call', 'tool_call_event', tool_call))
                    tools.append(tool_call)
    messages: List[Message] = []
    if content:
        messages.append({
            'role': 'ai',
            'text': ''.join(content),
        })
    if tools:
        messages.append({
            'role': 'tool_call',
            'tools': tools,
        })
    if len(messages) > 0:
        messages[-1] = _set_cost_raw(messages[-1], raw)
    return messages


def _wrap_event_callback_convert_ai_text_to_tool_call_text(
    wrapped: EventCallback,
) -> EventCallback:
    async def wrapper(event: EventPayload) -> None:
        if event[0] == 'ai' and event[1] == 'text_event':
            await wrapped(('tool_call', 'text_event', event[2]))
        else:
            await wrapped(event)

    return wrapper


def _get_ollama_format_for_structured_output(
    extraction_type: Type[ExtractionType],
) -> Dict:
    format: Dict = to_strict_json_schema(ensure_pydantic_model(extraction_type))

    docstr = getattr(extraction_type, '__doc__', None)
    if docstr:
        format['description'] = docstr

    return format


class LasagnaOllama(Model):
    def __init__(self, model: str, **model_kwargs: Dict[str, Any]):
        self.model = model
        self.model_kwargs = copy.deepcopy(model_kwargs or {})
        self.n_retries: int = cast(int, self.model_kwargs['retries']) if 'retries' in self.model_kwargs else 3
        if not isinstance(self.n_retries, int) or self.n_retries < 0:
            raise ValueError(f"model_kwargs['retries'] must be a non-negative integer (got {self.model_kwargs['retries']})")
        self.model_spec: ModelSpec = {
            'provider': 'ollama',
            'model': self.model,
            'model_kwargs': self.model_kwargs,
        }
        self.base_url = self.model_kwargs.get('base_url', os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434'))
        self.keep_alive = self.model_kwargs.get('keep_alive', '5m')
        self.timeout_seconds: float = cast(float, self.model_kwargs['timeout_seconds']) if 'timeout_seconds' in self.model_kwargs else 120.0
        if not isinstance(self.timeout_seconds, float) or self.timeout_seconds < 0.0:
            raise ValueError(f"model_kwargs['timeout_seconds'] must be a non-negative float (got {self.model_kwargs['timeout_seconds']})")
        self.payload_options = copy.deepcopy(self.model_kwargs)
        for key_to_remove in ['retries', 'base_url', 'keep_alive', 'timeout_seconds']:
            if key_to_remove in self.payload_options:
                del self.payload_options[key_to_remove]

    def config_hash(self) -> str:
        return recursive_hash(None, {
            'provider': 'ollama',
            'model': self.model,
            'model_kwargs': self.model_kwargs,
        })

    async def _run_once(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools_spec: Union[None, List[Dict]],
        force_tool: bool,
        format: Union[None, Dict],
    ) -> List[Message]:
        stream = True

        if tools_spec and format:
            raise ValueError("Oops! You cannot do both tool-use and structured output at the same time!")

        if tools_spec:
            stream = False   # Ollama does not yet support streaming tool responses.
            if not force_tool:
                raise ValueError("Oops! Ollama currently does not support *optional* tool use. Thus, if you pass tools, you must also pass `force_tool=True` to show that your intended use matches Ollama's behavior.")
        else:
            if force_tool:
                raise ValueError("Oops! You cannot force tools that are not specified!")

        ollama_messages = await _convert_to_ollama_messages(messages)

        _LOG.info(f"Invoking {self.model} with:\n  messages: {_log_dumps(ollama_messages)}\n  tools: {_log_dumps(tools_spec)}")

        url = f'{self.base_url}/api/chat'

        payload = {
            'model': self.model,
            'messages': ollama_messages,
            'stream': stream,
            **({'tools': tools_spec} if tools_spec else {}),
            **({'format': format} if format else {}),
            'options': self.payload_options,
            'keep_alive': self.keep_alive,
        }

        event_stream = _event_stream(url, payload, self.timeout_seconds)
        new_messages = await _process_stream(event_stream, event_callback)

        _LOG.info(f"Finished {self.model}")

        return new_messages

    async def _retrying_run_once(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools_spec: Union[None, List[Dict]],
        force_tool: bool,
        format: Union[None, Dict],
    ) -> List[Message]:
        last_error: Union[Exception, None] = None
        assert self.n_retries + 1 > 0   # <-- we know this is true from the check in __init__
        for delay_on_error in exponential_backoff_retry_delays(self.n_retries + 1):
            try:
                await event_callback(('transaction', 'start', ('ollama', self.model)))
                try:
                    new_messages = await self._run_once(
                        event_callback = event_callback,
                        messages = messages,
                        tools_spec = tools_spec,
                        force_tool = force_tool,
                        format = format,
                    )
                except:
                    await event_callback(('transaction', 'rollback', None))
                    raise
                await event_callback(('transaction', 'commit', None))
                return new_messages
            except Exception as e:
                # Some errors should be retried, some should not. Below
                # is the logic to decide when to retry vs when to not.
                # It's likely this will change as we get more usage and see
                # where Ollama tends to fail, and when we know more what is
                # recoverable vs not.
                last_error = e
                if isinstance(e, httpx.HTTPError):
                    # Network connection error. We can retry this.
                    pass
                else:
                    # Some other error that we don't know about. Let's bail.
                    raise
                if delay_on_error > 0.0:
                    _LOG.warning(f"Got a maybe-recoverable error (will retry in {delay_on_error:.2f} seconds): {e}")
                    await asyncio.sleep(delay_on_error)
        assert last_error is not None   # <-- we know this is true because `n_retries + 1 > 0`
        raise last_error

    async def run(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools: List[Callable],
        force_tool: bool = False,
        max_tool_iters: int = 5,
    ) -> List[Message]:
        messages = [*messages]  # shallow copy
        new_messages: List[Message] = []
        tools_spec = _convert_to_ollama_tools(tools)
        tools_map = {get_name(tool): tool for tool in tools}
        for _ in range(max_tool_iters):
            new_messages_here = await self._retrying_run_once(
                event_callback = event_callback,
                messages       = messages,
                tools_spec     = tools_spec,
                force_tool     = force_tool,
                format         = None,
            )
            tools_results = await handle_tools(
                prev_messages = messages,
                new_messages = new_messages_here,
                tools_map = tools_map,
                event_callback = event_callback,
                model_spec = self.model_spec,
            )
            new_messages.extend(new_messages_here)
            messages.extend(new_messages_here)
            if tools_results is None:
                break
            for tool_result in tools_results:
                await event_callback(('tool_res', 'tool_res_event', tool_result))
            tool_response_message = build_tool_response_message(tools_results)
            new_messages.append(tool_response_message)
            messages.append(tool_response_message)
        return new_messages

    async def extract(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        extraction_type: Type[ExtractionType],
    ) -> Tuple[Message, ExtractionType]:
        format = _get_ollama_format_for_structured_output(extraction_type)

        new_messages = await self._retrying_run_once(
            event_callback = _wrap_event_callback_convert_ai_text_to_tool_call_text(event_callback),
            messages       = messages,
            tools_spec     = None,
            force_tool     = False,
            format         = format,
        )

        assert len(new_messages) == 1
        new_message = new_messages[0]

        assert new_message['role'] == 'ai'  # Ollama generates structured output just like it does normal text, so at this point it is just text.

        arguments = new_message['text'] or '{}'

        tool_call: ToolCall = {
            'call_id': 'extraction_tool_call',
            'call_type': 'function',
            'function': {
                'name': get_name(extraction_type),
                'arguments': arguments,
            },
        }

        tool_message: MessageToolCall = {
            'role': 'tool_call',
            'tools': [tool_call],
        }
        if 'cost' in new_message:
            tool_message['cost'] = new_message['cost']
        if 'raw' in new_message:
            tool_message['raw'] = new_message['raw']

        await event_callback(('tool_call', 'tool_call_event', tool_call))

        parsed = json.loads(arguments)
        result = build_and_validate(extraction_type, parsed)

        return tool_message, result
