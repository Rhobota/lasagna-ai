from .types import (
    Message,
    MessageContent,
    MessageToolCall,
    EventCallback,
    EventPayload,
    Model,
    ToolCall,
    ToolParam,
    ToolResult,
    ModelRecord,
    Cost,
)

from typing import (
    List, Callable, AsyncIterator, Any, cast,
    Tuple, Dict, Optional, Union, Literal,
)

from .stream import (
    apeek,
    adup,
    prefix_stream,
)

from .util import (
    parse_docstring,
    combine_pairs,
    convert_to_image_base64,
    exponential_backoff_retry_delays,
    recursive_hash,
)

from .tools import handle_tools, build_tool_response_message

from anthropic import AsyncAnthropic, NOT_GIVEN

from anthropic.types import MessageParam
from anthropic.types.message import Message as AnthropicMessage
from anthropic.lib.streaming._types import MessageStreamEvent
from anthropic.types.content_block import ContentBlock
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.image_block_param import ImageBlockParam
from anthropic.types.tool_use_block_param import ToolUseBlockParam
from anthropic.types.tool_result_block_param import ToolResultBlockParam

import asyncio
import copy
import json
import inspect

import logging

_LOG = logging.getLogger(__name__)


ANTHROPIC_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'claude-3-opus-20240229',
        'display_name': 'Claude3 Opus',
    },
    {
        'formal_name': 'claude-3-sonnet-20240229',
        'display_name': 'Claude3 Sonnet',
    },
    {
        'formal_name': 'claude-3-haiku-20240307',
        'display_name': 'Claude3 Haiku',
    },
]


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
            if m['media_type'] == 'image':
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
                raise ValueError(f"unknown media type: {m['media_type']}")
    if len(ret) == 0:
        raise ValueError(f"no content")
    return ret


async def _convert_to_anthropic_messages(
    messages: List[Message],
) -> Tuple[Union[str, None], List[MessageParam]]:
    system_prompt: Union[str, None] = None
    ret: List[MessageParam] = []
    if len(messages) > 0:
        first_message = messages[0]
        if first_message['role'] == 'system':
            if first_message.get('media'):
                raise ValueError(f"You may not pass media in the system prompt.")
            system_prompt = first_message['text']
            messages = messages[1:]
        for m in messages:
            if m['role'] == 'system':
                raise ValueError(f"You can only have a system prompt as the first message!")
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
                pass
            elif m['role'] == 'tool_res':
                pass
            else:
                raise ValueError(f"Unknown role: {m['role']}")
    return system_prompt, ret


async def _process_stream(stream: AsyncIterator[MessageStreamEvent]) -> AsyncIterator[EventPayload]:
    async for event in stream:
        # TODO
        e: EventPayload = ('ai', 'text_event', '')
        yield e


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
            pass  # TODO
        else:
            raise ValueError(f"unknown content type: {c.type}")
    if len(ms) == 0:
        raise ValueError("no content")
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


def _log_dumps(val: Any) -> str:
    if isinstance(val, dict):
        return json.dumps(val)
    else:
        return str(val)


class LasagnaAnthropic(Model):
    def __init__(self, model: str, **model_kwargs: Dict[str, Any]):
        known_model_names = [m['formal_name'] for m in ANTHROPIC_KNOWN_MODELS]
        if model not in known_model_names:
            raise ValueError(f'unknown model: {model}')
        self.model = model
        self.model_kwargs = copy.deepcopy(model_kwargs or {})
        self.n_retries: int = cast(int, self.model_kwargs['retries']) if 'retries' in self.model_kwargs else 3
        if not isinstance(self.n_retries, int) or self.n_retries < 0:
            raise ValueError(f"model_kwargs['retries'] must be a non-negative integer (got {self.model_kwargs['retries']})")

    def config_hash(self) -> str:
        return recursive_hash(None, {
            'provider': 'anthropic',
            'model': self.model,
            'model_kwargs': self.model_kwargs,
        })

    async def _run_once(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools: List[Callable],
        force_tool: bool,
    ) -> List[Message]:
        # TODO use `force_tool`

        #tools_spec = _convert_to_anthropic_tools(tools)

        system_prompt, anthropic_messages = await _convert_to_anthropic_messages(messages)

        # TODO use self.model_kwargs

        #_LOG.info(f"Invoking {self.model} with:\n  system_prompt: {system_prompt}\n  messages: {_log_dumps(anthropic_messages)}\n  tools: {_log_dumps(tools_spec)}\n  tool_choice: {tool_choice}")
        _LOG.info(f"Invoking {self.model} with:\n  system_prompt: {system_prompt}\n  messages: {_log_dumps(anthropic_messages)}")

        client = AsyncAnthropic()
        async with client.messages.stream(
            model    = self.model,
            system   = system_prompt or NOT_GIVEN,
            messages = anthropic_messages,
            max_tokens = 1024,
            # logprobs?
        ) as stream:
            raw_stream, rt_stream = adup(stream)
            async for event in _process_stream(rt_stream):
                await event_callback(event)
            raw_events = [v.to_dict() async for v in raw_stream]
            anthropic_message = await stream.get_final_message()

        new_messages = _build_messages_from_anthropic_payload(raw_events, anthropic_message)

        _LOG.info(f"Finished {self.model} with usage: {_log_dumps(new_messages[-1].get('cost'))}")

        return new_messages

    # TODO: retry method

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
        for _ in range(max_tool_iters):
            new_messages_here = await self._run_once(
                event_callback = event_callback,
                messages       = messages,
                tools          = tools,
                force_tool     = force_tool,
            )
            tools_map = {tool.__name__: tool for tool in tools}
            new_messages.extend(new_messages_here)
            messages.extend(new_messages_here)
            tools_results = await handle_tools(new_messages_here, tools_map)
            if tools_results is None:
                break
            for tool_result in tools_results:
                await event_callback(('tool_res', 'tool_res_event', tool_result))
            tool_response_message = build_tool_response_message(tools_results)
            new_messages.append(tool_response_message)
            messages.append(tool_response_message)
        return new_messages
