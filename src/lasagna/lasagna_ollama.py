"""
This module is the Lasagna adapter for the Ollama server.

For more information about Ollama, see:
 - https://ollama.com/
"""

from .types import (
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

from .known_models import OLLAMA_KNOWN_MODELS

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
        'name': get_name(tool),
        'description': description,
        'input_schema': convert_to_json_schema(params),
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
    # Ollama only supports *images* as media. But, so does lasagna, so all good.
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
        elif m['role'] == 'tool_res':
            for t in m['tools']:
                # t['is_error'] IS NOT USED!
                res.append({
                    'role': 'tool',
                    'content': extract_tool_result_as_sting(t),
                    'name': t['call_id'],  # <-- weird, I know, but Ollama sort of uses the name as the call id, so we do that too
                })
        else:
            raise RuntimeError(f"unreachable: {m['role']}")
    return res


async def _event_stream(url: str, payload: Dict) -> AsyncIterator[Dict]:
    async with httpx.AsyncClient() as client:
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


async def _process_stream(
    stream: AsyncIterator[Dict],
    event_callback: EventCallback,
) -> List[Message]:
    async for event in stream:
        if 'message' in event:
            m = event['message']
            if 'content' in m:
                c = m['content']
                assert isinstance(c, str)
                await event_callback(('ai', 'text_event', c))  # TODO
    return []  # TODO


class LasagnaOllama(Model):
    def __init__(self, model: str, **model_kwargs: Dict[str, Any]):
        known_model_names = [m['formal_name'] for m in OLLAMA_KNOWN_MODELS]
        if model not in known_model_names:
            _LOG.warning(f'untested model: {model} (may or may not work)')
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
        self.base_url = model_kwargs.get('base_url', os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434'))

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
    ) -> List[Message]:
        stream = True
        tools = None

        if tools_spec and len(tools_spec) > 0:
            stream = False   # Ollama does not yet support streaming tool responses.
            tools = tools_spec
            if not force_tool:
                raise ValueError("Oops! Ollama currently does not support *optional* tool use. Thus, if you pass tools, you must also pass `force_tool=True` to show that your intended use matches Ollama's behavior.")

        ollama_messages = await _convert_to_ollama_messages(messages)

        _LOG.info(f"Invoking {self.model} with:\n  messages: {_log_dumps(ollama_messages)}\n  tools: {_log_dumps(tools)}")

        url = f'{self.base_url}/api/chat'

        payload = {
            'model': self.model,
            'messages': ollama_messages,
            'stream': stream,
            **({'tools': tools} if tools is not None else {}),
        }

        event_stream = _event_stream(url, payload)
        new_messages = await _process_stream(event_stream, event_callback)

        _LOG.info(f"Finished {self.model}")

        return new_messages

    async def _retrying_run_once(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools_spec: Union[None, List[Dict]],
        force_tool: bool,
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
        tools_spec: List[Dict] = [
            {
                'name': get_name(extraction_type),
                'input_schema': to_strict_json_schema(ensure_pydantic_model(extraction_type)),
            },
        ]

        # TODO: use the `format` feature of Ollama

        docstr = getattr(extraction_type, '__doc__', None)
        if docstr:
            tools_spec[0]['description'] = docstr

        new_messages = await self._retrying_run_once(
            event_callback = event_callback,
            messages       = messages,
            tools_spec     = tools_spec,
            force_tool     = True,
        )

        assert len(new_messages) == 1
        new_message = new_messages[0]

        if new_message['role'] == 'tool_call':
            tools = new_message['tools']

            assert len(tools) == 1
            parsed = json.loads(tools[0]['function']['arguments'])
            result = build_and_validate(extraction_type, parsed)

            return new_message, result

        else:
            assert new_message['role'] == 'ai'
            text = new_message['text']
            raise RuntimeError(f"Model failed to generate structured output; instead, it output: {text}")
