from .types import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageRole,
    EventCallback,
    LLM,
    ToolCall,
    ToolParam,
)

from .stream import (
    apeek,
    adup,
    prefix_stream,
)

from .util import parse_docstring

from openai import AsyncOpenAI, NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from typing import (
    List, Callable, AsyncIterator, Any, Tuple, Dict, Optional, Union,
)

import copy
import json


async def _process_text_stream(
    stream: AsyncIterator[Tuple[ChoiceDelta, Any]],
) -> AsyncIterator[Tuple[ChatMessageRole, str, Union[str, ToolCall]]]:
    role = ChatMessageRole.AI
    async for delta, finish_reason in stream:
        text = delta.content
        if finish_reason is not None:
            if text is not None:
                yield role, 'text', str(text)
            return
        if text is None:
            # The model is switching from text to tools!
            yield role, 'text', "\n\n"
            fixed_stream = prefix_stream([(delta, finish_reason)], stream)
            substream = _process_tool_call_stream(fixed_stream)
            async for subval in substream:
                yield subval
            return
        yield role, 'text', str(text)


async def _process_tool_call_stream(
    stream: AsyncIterator[Tuple[ChoiceDelta, Any]],
) -> AsyncIterator[Tuple[ChatMessageRole, str, Union[str, ToolCall]]]:
    role = ChatMessageRole.TOOL_CALL
    recs_by_index: Dict[int, ToolCall] = {}
    args_by_index: Dict[int, List[str]] = {}
    last_index: Optional[int] = None
    async for delta, _ in stream:
        if not delta.tool_calls:
            continue
        for tc in delta.tool_calls:
            index = tc.index
            if index != last_index and last_index is not None:
                yield role, 'text_tool', ")\n"   # <-- again, assumes no index-interleave
            last_index = index
            if index not in recs_by_index:
                assert tc.type == 'function', f"The only tool type we can do is a function! But got: {tc.type}"
                assert tc.function
                assert tc.function.name
                assert tc.id
                n: str = tc.function.name
                a: str = tc.function.arguments or ''
                recs_by_index[index] = {
                    'call_id': tc.id,
                    'call_type': 'function',
                    'function': {
                        'name': n,
                        'arguments': '',  # will be filled in at the end
                    },
                }
                args_by_index[index] = [a]
                yield role, 'text_tool', f"{n}("   # <-- assumes no index-interleave
            else:
                # assumes nothing but the argument is in the delta message...
                args = args_by_index[index]
                assert tc.function
                assert tc.function.arguments
                assert not tc.function.name
                a_delta: str = tc.function.arguments
                args.append(a_delta)
                yield role, 'text_tool', a_delta
    if last_index is not None:
        yield role, 'text_tool', ")"   # <-- again, assumes no index-interleave
    for index in sorted(recs_by_index.keys()):
        rec = recs_by_index[index]
        rec['function']['arguments'] = ''.join(args_by_index[index])
        yield role, 'function_call', rec


async def _process_output_stream(stream: AsyncIterator[Tuple[ChoiceDelta, Any]]) -> AsyncIterator[Tuple[ChatMessageRole, str, Union[str, ToolCall]]]:
    first, stream = await apeek(stream, n=1)
    first_delta, _ = first[0]
    is_text = first_delta.content is not None   # <-- hacky, but works?
    if is_text:
        gen = _process_text_stream
    else:
        gen = _process_tool_call_stream
    async for v in gen(stream):
        yield v


async def _extract_deltas(stream: AsyncIterator[ChatCompletionChunk]) -> AsyncIterator[Tuple[ChoiceDelta, Any]]:
    async for v in stream:
        assert len(v.choices) == 1, f"Why do we have {len(v.choices)} choices?"
        single_choice = v.choices[0]
        yield single_choice.delta, single_choice.finish_reason


def _convert_to_json_schema(params: List[ToolParam]) -> Dict[str, object]:
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
                "description": p['description'],
            }
            for p in params
        },
        "required": [
            p['name']
            for p in params
            if not p['description'].startswith('(optional)')
        ],
    }


def _convert_to_openai_tool(tool: Callable) -> ChatCompletionToolParam:
    description, params = parse_docstring(tool.__doc__ or '')
    return {
        'type': 'function',
        'function': {
            'name': tool.__name__,
            'description': description,
            'parameters': _convert_to_json_schema(params),
        },
    }


def _convert_to_openai_tools(tools: List[Callable]) -> Union[NotGiven, List[ChatCompletionToolParam]]:
    if len(tools) == 0:
        return NOT_GIVEN
    specs = [_convert_to_openai_tool(tool) for tool in tools]
    return specs


def _convert_to_openai_messages(messages: List[ChatMessage]) -> List[ChatCompletionMessageParam]:
    ms: List[ChatCompletionMessageParam] = []
    for m in messages:
        if m['role'] == ChatMessageRole.TOOL_CALL:
            tool_calls = m['tools']
            ms.append({
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': t['call_id'],
                        'type': t['call_type'],
                        'function': {
                            'name': t['function']['name'],
                            'arguments': t['function']['arguments'],
                        },
                    }
                    for t in tool_calls
                ],
            })
        elif m['role'] == ChatMessageRole.TOOL_RES:
            tool_results = m['tools']
            for t in tool_results:
                ms.append({
                    'role': 'tool',
                    'content': str(t['result']),
                    'tool_call_id': t['call_id'],
                })
        elif m['role'] == ChatMessageRole.SYSTEM:
            ms.append({
                'role': 'system',
                'content': m['text'],
            })
        elif m['role'] == ChatMessageRole.HUMAN:
            ms.append({
                'role': 'user',
                'content': m['text'],
            })
        elif m['role'] == ChatMessageRole.AI:
            ms.append({
                'role': 'assistant',
                'content': m['text'],
            })
        else:
            raise ValueError(f"unknown message role: {m['role']}")
    # TODO: collapse subsequent messages that are both 'assistant'
    return ms


def _build_messages_from_openai_payload(
    payload: List[ChatCompletionChunk],
    events: List[Tuple[ChatMessageRole, str, Union[str, ToolCall]]],
) -> List[ChatMessage]:
    """
    We either have:
     - all AI events
     - all TOOL_CALL events
     - some AI events then it switches to TOOL_CALL events
    """
    ai_events = [
        event
        for event in events
        if event[0] == ChatMessageRole.AI
    ]
    tool_events = [
        event
        for event in events
        if event[0] == ChatMessageRole.TOOL_CALL
    ]
    ai_message: Optional[ChatMessage] = {
        'role': ChatMessageRole.AI,
        'text': ''.join([e[2] for e in ai_events if e[1] == 'text' and isinstance(e[2], str)]),
        'cost': None,  # TODO: OpenAI's API doesn't return this info in streaming mode! Hopefully they will in the future.
        'raw': payload,
    } if len(ai_events) > 0 else None
    tool_message: Optional[ChatMessageToolCall] = {
        'role': ChatMessageRole.TOOL_CALL,
        'tools': [e[2] for e in tool_events if e[1] == 'function_call' and not isinstance(e[2], str)],
        'cost': None,  # TODO: OpenAI's API doesn't return this info in streaming mode! Hopefully they will in the future.
        'raw': payload,
    } if len(tool_events) > 0 else None
    if ai_message and tool_message:
        ai_message['cost'] = None  # <-- we don't want to double-count this
        ai_message['raw'] = None
        return [ai_message, tool_message]
    elif ai_message:
        return [ai_message]
    elif tool_message:
        return [tool_message]
    else:
        raise ValueError('no events')


