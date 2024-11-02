"""
This module is the Lasagna adapter for the Anthropic models.

For more information about the Anthropic models this adapter is for, see:
 - https://docs.anthropic.com/en/docs/welcome
"""

from .types import (
    ModelSpec,
    Message,
    MessageContent,
    MessageToolCall,
    MessageToolResult,
    EventCallback,
    EventPayload,
    ToolCall,
    Model,
    Cost,
    ExtractionType,
)

from .stream import (
    adup,
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

from .known_models import ANTHROPIC_KNOWN_MODELS

from anthropic import (
    AsyncAnthropic,
    NOT_GIVEN,
    NotGiven,
    AnthropicError,
    APIConnectionError,
    APIStatusError,
)
from anthropic.types import MessageParam, ToolParam as AnthropicToolParam
from anthropic.types.message import Message as AnthropicMessage
from anthropic.types.message_create_params import ToolChoice
from anthropic.lib.streaming._types import MessageStreamEvent
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.image_block_param import ImageBlockParam
from anthropic.types.tool_use_block_param import ToolUseBlockParam
from anthropic.types.tool_result_block_param import ToolResultBlockParam

from openai.lib._pydantic import to_strict_json_schema

from typing import (
    List, Callable, AsyncIterator, Any, Type,
    Tuple, Dict, Union, Literal,
    cast,
)

import asyncio
import copy
import json

import logging

_LOG = logging.getLogger(__name__)


async def _build_anthropic_content(
    message: MessageContent,
) -> List[Union[TextBlockParam, ImageBlockParam]]:
    ret: List[Union[TextBlockParam, ImageBlockParam]] = []
    if message['text']:
        ret.append({
            'type': 'text',
            'text': message['text'],
        })
    if 'media' in message:
        for m in message['media']:
            if m['type'] == 'image':
                mimetype, data = await convert_to_image_base64(m['image'])
                ret.append({
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': mimetype,
                        'data': data,
                    },
                })
            else:
                raise ValueError(f"unknown media type: {m['type']}")
    return ret


async def _build_anthropic_tool_use(
    message: MessageToolCall,
) -> List[ToolUseBlockParam]:
    ret: List[ToolUseBlockParam] = []
    for tool in message['tools']:
        if tool['call_type'] == 'function':
            ret.append({
                'type': 'tool_use',
                'id': tool['call_id'],
                'name': tool['function']['name'],
                'input': json.loads(tool['function']['arguments']),
            })
        else:
            raise ValueError(f"unknown tool type: {tool['call_type']}")
    if len(ret) == 0:
        raise ValueError(f"no content")
    return ret


async def _build_anthropic_tool_result(
    message: MessageToolResult,
) -> List[ToolResultBlockParam]:
    ret: List[ToolResultBlockParam] = []
    for tool_result in message['tools']:
        content = extract_tool_result_as_sting(tool_result)
        obj: ToolResultBlockParam = {
            'type': 'tool_result',
            'tool_use_id': tool_result['call_id'],
            'content': [
                {
                    'type': 'text',
                    'text': content,
                },
            ],
        }
        if 'is_error' in tool_result and tool_result['is_error']:
            obj['is_error'] = True
        ret.append(obj)
    if len(ret) == 0:
        raise ValueError(f"no content")
    return ret


def _collapse_anthropic_messages(
    messages: List[MessageParam],
) -> List[MessageParam]:
    ms: List[MessageParam] = []
    for i, m in enumerate(messages):
        if i == 0:
            ms.append(m)
            continue
        prev_m = ms[-1]
        if prev_m['role'] == m['role']:
            assert isinstance(prev_m['content'], list)
            assert isinstance(m['content'], list)
            prev_m['content'].extend(m['content'])
        else:
            ms.append(m)
    return ms


def _collapse_tool_call_messages(
    messages: List[Message],
) -> List[Message]:
    ms: List[Message] = []
    for i, m in enumerate(messages):
        if i == 0:
            ms.append(m)
            continue
        prev_m = ms[-1]
        if prev_m['role'] == 'tool_call' and m['role'] == 'tool_call':
            prev_m['tools'].extend(m['tools'])
        else:
            ms.append(m)
    return ms


async def _convert_to_anthropic_messages(
    messages: List[Message],
) -> Tuple[Union[str, None], List[MessageParam]]:
    system_prompt: Union[str, None] = None
    ret: List[MessageParam] = []
    if len(messages) > 0:
        first_message = messages[0]
        if first_message['role'] == 'system':
            if first_message.get('media'):
                raise ValueError(f"For this model, you may not pass media in the system prompt.")
            system_prompt = first_message['text']
            messages = messages[1:]
        for m in messages:
            if m['role'] == 'system':
                raise ValueError(f"For this model, you can only have a system prompt as the first message!")
            elif m['role'] == 'ai':
                ret.append({
                    'role': 'assistant',
                    'content': await _build_anthropic_content(m),
                })
            elif m['role'] == 'human':
                ret.append({
                    'role': 'user',
                    'content': await _build_anthropic_content(m),
                })
            elif m['role'] == 'tool_call':
                ret.append({
                    'role': 'assistant',
                    'content': await _build_anthropic_tool_use(m),
                })
            elif m['role'] == 'tool_res':
                ret.append({
                    'role': 'user',
                    'content': await _build_anthropic_tool_result(m),
                })
            else:
                raise ValueError(f"Unknown role: {m['role']}")
    ret = [r for r in ret if len(list(r['content'])) > 0]
    ret = _collapse_anthropic_messages(ret)
    return system_prompt, ret


def _build_messages_from_anthropic_payload(
    raw_events: List[Dict[str, Any]],
    message: AnthropicMessage,
) -> List[Message]:
    ms: List[Message] = []
    for c in message.content:
        if c.type == 'text':
            ms.append({
                'role': 'ai',
                'text': c.text,
            })
        elif c.type == 'tool_use':
            ms.append({
                'role': 'tool_call',
                'tools': [{
                    'call_id': c.id,
                    'call_type': 'function',
                    'function': {
                        'name': c.name,
                        'arguments': json.dumps(c.input),
                    },
                }],
            })
        else:
            raise ValueError(f"unknown content type: {c.type}")
    ms = _collapse_tool_call_messages(ms)
    if len(ms) == 0:
        ms.append({
            'role': 'ai',
            'text': None,  # <-- sometimes it generates nothing, so we'll show that
        })
    last_message = ms[-1]
    cost: Cost = {
        'input_tokens': message.usage.input_tokens,
        'output_tokens': message.usage.output_tokens,
        'total_tokens': message.usage.input_tokens + message.usage.output_tokens,
    }
    last_message['cost'] = cost
    last_message['raw'] = {
        'events': raw_events,
        'message': message.to_dict(),
    }
    return ms


def _convert_to_anthropic_tool(tool: Callable) -> AnthropicToolParam:
    description, params = get_tool_params(tool)
    return {
        'name': get_name(tool),
        'description': description,
        'input_schema': convert_to_json_schema(params),
    }


def _convert_to_anthropic_tools(tools: List[Callable]) -> Union[NotGiven, List[AnthropicToolParam]]:
    if len(tools) == 0:
        return NOT_GIVEN
    specs = [_convert_to_anthropic_tool(tool) for tool in tools]
    return specs


def _make_raw_event(event: MessageStreamEvent) -> Dict[str, Any]:
    d = event.to_dict()
    if 'snapshot' in d:
        # This accumulated text is overkill.
        # It is O(n^2) in storage, so we don't want this in our database.
        del d['snapshot']
    return d


async def _process_stream(stream: AsyncIterator[MessageStreamEvent]) -> AsyncIterator[EventPayload]:
    content: Dict[int, Tuple[Literal['text', 'tool_use'], str, List[str]]] = {}
    async for event in stream:
        if event.type == 'content_block_start':
            assert event.index not in content
            if event.content_block.type == 'text':
                content[event.index] = event.content_block.type, '', [event.content_block.text]
                if event.content_block.text:
                    yield 'ai', 'text_event', event.content_block.text
            elif event.content_block.type == 'tool_use':
                assert event.content_block.input == {}
                content[event.index] = event.content_block.type, event.content_block.name, []
                yield 'tool_call', 'text_event', f'{event.content_block.name}('
            else:
                raise ValueError(f"unknown content block type: {event.content_block.type}")
        elif event.type == 'content_block_delta':
            assert event.index in content
            if event.delta.type == 'text_delta':
                content[event.index][2].append(event.delta.text)
                if event.delta.text:
                    yield 'ai', 'text_event', event.delta.text
            elif event.delta.type == 'input_json_delta':
                content[event.index][2].append(event.delta.partial_json)
                if event.delta.partial_json:
                    yield 'tool_call', 'text_event', event.delta.partial_json
            else:
                raise ValueError(f"unknown delta type: {event.delta.type}")
        elif event.type == 'content_block_stop':
            assert event.index in content
            if event.content_block.type == 'text':
                pass  # nothing more to do... we already streamed all the text
            elif event.content_block.type == 'tool_use':
                yield 'tool_call', 'text_event', ')\n'
                tool_call: ToolCall = {
                    'call_type': 'function',
                    'call_id': event.content_block.id,
                    'function': {
                        'name': event.content_block.name,
                        'arguments': json.dumps(event.content_block.input),
                    },
                }
                yield 'tool_call', 'tool_call_event', tool_call
            else:
                raise ValueError(f"unknown content block type: {event.content_block.type}")
        else:
            # The other events we can ignore, I think.
            pass


def _log_dumps(val: Any) -> str:
    if isinstance(val, dict):
        return json.dumps(val)
    else:
        return str(val)


class LasagnaAnthropic(Model):
    def __init__(self, model: str, **model_kwargs: Dict[str, Any]):
        known_model_names = [m['formal_name'] for m in ANTHROPIC_KNOWN_MODELS]
        if model not in known_model_names:
            _LOG.warning(f'untested model: {model} (may or may not work)')
        self.model = model
        self.model_kwargs = copy.deepcopy(model_kwargs or {})
        self.n_retries: int = cast(int, self.model_kwargs['retries']) if 'retries' in self.model_kwargs else 3
        if not isinstance(self.n_retries, int) or self.n_retries < 0:
            raise ValueError(f"model_kwargs['retries'] must be a non-negative integer (got {self.model_kwargs['retries']})")
        self.model_spec: ModelSpec = {
            'provider': 'anthropic',
            'model': self.model,
            'model_kwargs': self.model_kwargs,
        }

    def config_hash(self) -> str:
        return recursive_hash(None, {
            'provider': 'anthropic',
            'model': self.model,
            'model_kwargs': self.model_kwargs,
        })

    def _make_client(self) -> AsyncAnthropic:
        api_key: Union[str, None] = cast(str, self.model_kwargs['api_key']) if 'api_key' in self.model_kwargs else None
        base_url: Union[str, None] = cast(str, self.model_kwargs['base_url']) if 'base_url' in self.model_kwargs else None
        client = AsyncAnthropic(
            api_key  = api_key,
            base_url = base_url,
        )
        return client

    async def _run_once(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools_spec: Union[NotGiven, List[AnthropicToolParam]],
        force_tool: bool,
        disable_parallel_tool_use: bool,
    ) -> List[Message]:
        tool_choice: Union[NotGiven, ToolChoice]
        if force_tool:
            if not tools_spec or len(tools_spec) == 0:
                raise ValueError(f"When `force_tool` is set, you must pass at least one tool!")
            elif len(tools_spec) == 1:
                tool_choice = {
                    "type": "tool",
                    "name": tools_spec[0]['name'],
                    "disable_parallel_tool_use": disable_parallel_tool_use,
                }
            else:
                tool_choice = {
                    "type": "any",   # <-- model must use a tool, but is allowed to choose which one on its own
                    "disable_parallel_tool_use": disable_parallel_tool_use,
                }
        else:
            if tools_spec and len(tools_spec) > 0:
                tool_choice = {
                    "type": "auto",  # <-- the model can choose to use them or not
                    "disable_parallel_tool_use": disable_parallel_tool_use,
                }
            else:
                tool_choice = NOT_GIVEN

        system_prompt, anthropic_messages = await _convert_to_anthropic_messages(messages)

        max_tokens: int = cast(int, self.model_kwargs['max_tokens']) if 'max_tokens' in self.model_kwargs else 4096
        stop: Union[List[str], NotGiven] = cast(List[str], self.model_kwargs['stop']) if 'stop' in self.model_kwargs else NOT_GIVEN
        temperature: Union[float, NotGiven] = cast(float, self.model_kwargs['temperature']) if 'temperature' in self.model_kwargs else NOT_GIVEN
        top_p: Union[float, NotGiven] = cast(float, self.model_kwargs['top_p']) if 'top_p' in self.model_kwargs else NOT_GIVEN
        top_k: Union[int, NotGiven] = cast(int, self.model_kwargs['top_k']) if 'top_k' in self.model_kwargs else NOT_GIVEN
        user: Union[str, None] = cast(str, self.model_kwargs['user']) if 'user' in self.model_kwargs else None

        _LOG.info(f"Invoking {self.model} with:\n  system_prompt: {system_prompt}\n  messages: {_log_dumps(anthropic_messages)}\n  tools: {_log_dumps(tools_spec)}\n  tool_choice: {tool_choice}")

        client = self._make_client()
        async with client.messages.stream(
            model       = self.model,
            system      = system_prompt or NOT_GIVEN,
            messages    = anthropic_messages,
            max_tokens  = max_tokens,
            tools       = tools_spec,
            tool_choice = tool_choice,
            temperature = temperature,
            top_p       = top_p,
            top_k       = top_k,
            metadata    = {
                'user_id': user,
            },
            stop_sequences = stop,
            # Anthropic doesn't support sending logprobs 👎
        ) as stream:
            raw_stream, rt_stream = adup(stream)
            async for event in _process_stream(rt_stream):
                await event_callback(event)
            raw_events = [_make_raw_event(v) async for v in raw_stream]
            anthropic_message = await stream.get_final_message()
            new_messages = _build_messages_from_anthropic_payload(raw_events, anthropic_message)

        _LOG.info(f"Finished {self.model} with usage: {_log_dumps(new_messages[-1].get('cost'))}")

        return new_messages

    async def _retrying_run_once(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools_spec: Union[NotGiven, List[AnthropicToolParam]],
        force_tool: bool,
        disable_parallel_tool_use: bool,
    ) -> List[Message]:
        last_error: Union[Exception, None] = None
        assert self.n_retries + 1 > 0   # <-- we know this is true from the check in __init__
        for delay_on_error in exponential_backoff_retry_delays(self.n_retries + 1):
            try:
                await event_callback(('transaction', 'start', ('anthropic', self.model)))
                try:
                    new_messages = await self._run_once(
                        event_callback = event_callback,
                        messages = messages,
                        tools_spec = tools_spec,
                        force_tool = force_tool,
                        disable_parallel_tool_use = disable_parallel_tool_use,
                    )
                except:
                    await event_callback(('transaction', 'rollback', None))
                    raise
                await event_callback(('transaction', 'commit', None))
                return new_messages
            except AnthropicError as e:
                # Some errors should be retried, some should not. Below
                # is the logic to decide when to retry vs when to not.
                # It's likely this will change as we get more usage and see
                # where Anthropic tends to fail, and when we know more what is
                # recoverable vs not.
                last_error = e
                if isinstance(e, APIConnectionError):
                    # Network connection error. We can retry this.
                    pass
                elif isinstance(e, APIStatusError):
                    if 400 <= e.status_code < 500:
                        # This is a request error. Not recoverable.
                        raise
                    elif e.status_code >= 500:
                        # Server error, so we can retry this.
                        pass
                    else:
                        # Something else??? This is weird, maybe this is recoverable?
                        _LOG.warning(f"Got an unknown error status: {e}")
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
        tools_spec = _convert_to_anthropic_tools(tools)
        tools_map = {get_name(tool): tool for tool in tools}
        for _ in range(max_tool_iters):
            new_messages_here = await self._retrying_run_once(
                event_callback = event_callback,
                messages       = messages,
                tools_spec     = tools_spec,
                force_tool     = force_tool,
                disable_parallel_tool_use = False,
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
        tools_spec: List[AnthropicToolParam] = [
            {
                'name': get_name(extraction_type),
                'input_schema': to_strict_json_schema(ensure_pydantic_model(extraction_type)),
            },
        ]

        docstr = getattr(extraction_type, '__doc__', None)
        if docstr:
            tools_spec[0]['description'] = docstr

        new_messages = await self._retrying_run_once(
            event_callback = event_callback,
            messages       = messages,
            tools_spec     = tools_spec,
            force_tool     = True,
            disable_parallel_tool_use = True,
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
