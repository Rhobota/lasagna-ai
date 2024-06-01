from .types import (
    ChatMessage,
    ChatMessageContent,
    ChatMessageToolCall,
    ChatMessageRole,
    EventCallback,
    EventPayload,
    LLM,
    ToolCall,
    ToolParam,
    ToolResult,
    ModelRecord,
    Cost,
)

from .stream import (
    apeek,
    adup,
    prefix_stream,
)

from .util import (
    parse_docstring,
    combine_pairs,
    convert_to_image_url,
)

from openai import AsyncOpenAI, NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionMessageParam,
    ChatCompletionContentPartParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from typing import (
    List, Callable, AsyncIterator, Any, cast,
    Tuple, Dict, Optional, Union, Literal,
)

import asyncio
import copy
import json
import inspect


OPENAI_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'gpt-4o-2024-05-13',
        'display_name': 'GPT-4o',
    },
    {
        'formal_name': 'gpt-4-turbo-2024-04-09',
        'display_name': 'GPT-4',
    },
    {
        'formal_name': 'gpt-3.5-turbo-0125',
        'display_name': 'GPT-3.5',
    },
]


async def _process_text_stream(
    stream: AsyncIterator[Tuple[ChoiceDelta, Union[str, None]]],
) -> AsyncIterator[EventPayload]:
    async for delta, finish_reason in stream:
        text = delta.content
        if finish_reason is not None:
            if text is not None:
                yield ChatMessageRole.AI, 'text_event', str(text)
            return
        if text is None:
            # The model is switching from text to tools!
            yield ChatMessageRole.AI, 'text_event', "\n\n"
            put_back_val: Tuple[ChoiceDelta, Union[str, None]] = (delta, finish_reason)
            fixed_stream = prefix_stream([put_back_val], stream)
            substream = _process_tool_call_stream(fixed_stream)
            async for subval in substream:
                yield subval
            return
        yield ChatMessageRole.AI, 'text_event', str(text)


async def _process_tool_call_stream(
    stream: AsyncIterator[Tuple[ChoiceDelta, Union[str, None]]],
) -> AsyncIterator[EventPayload]:
    recs_by_index: Dict[int, ToolCall] = {}
    args_by_index: Dict[int, List[str]] = {}
    last_index: Optional[int] = None
    async for delta, _ in stream:
        if not delta.tool_calls:
            continue
        for tc in delta.tool_calls:
            index = tc.index
            if index != last_index and last_index is not None:
                yield ChatMessageRole.TOOL_CALL, 'text_event', ")\n"   # <-- again, assumes no index-interleave
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
                yield ChatMessageRole.TOOL_CALL, 'text_event', f"{n}("   # <-- assumes no index-interleave
            else:
                # assumes nothing but the argument is in the delta message...
                args = args_by_index[index]
                assert tc.function
                assert tc.function.arguments
                assert not tc.function.name
                a_delta: str = tc.function.arguments
                args.append(a_delta)
                yield ChatMessageRole.TOOL_CALL, 'text_event', a_delta
    if last_index is not None:
        yield ChatMessageRole.TOOL_CALL, 'text_event', ")"   # <-- again, assumes no index-interleave
    for index in sorted(recs_by_index.keys()):
        rec = recs_by_index[index]
        rec['function']['arguments'] = ''.join(args_by_index[index])
        yield ChatMessageRole.TOOL_CALL, 'tool_call_event', rec


async def _process_output_stream(
    stream: AsyncIterator[Tuple[ChoiceDelta, Union[str, None]]],
) -> AsyncIterator[EventPayload]:
    first, stream = await apeek(stream, n=1)
    first_delta, _ = first[0]
    is_text = first_delta.content is not None   # <-- hacky, but works?
    if is_text:
        gen = _process_text_stream
    else:
        gen = _process_tool_call_stream
    async for v in gen(stream):
        yield v


async def _extract_deltas(
    stream: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[Tuple[ChoiceDelta, Union[str, None]]]:
    async for v in stream:
        if len(v.choices) == 0:
            # The final message that has the `usage` has zero choices.
            # So just skip it here!
            continue
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


async def _make_openai_content(
    message: ChatMessageContent,
) -> List[ChatCompletionContentPartParam]:
    ret: List[ChatCompletionContentPartParam] = []
    if message['text']:
        ret.append({
            'type': 'text',
            'text': message['text'],
        })
    if message['media']:
        for m in message['media']:
            if m['media_type'] == 'image':
                ret.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': (await convert_to_image_url(m['image'])),
                    },
                })
            else:
                raise ValueError(f"unknown media type: {m['media_type']}")
    if len(ret) == 0:
        raise ValueError('no content in this message!')
    return ret


async def _convert_to_openai_messages(messages: List[ChatMessage]) -> List[ChatCompletionMessageParam]:
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
            if m['media']:
                raise ValueError('This model does not support media in the system prompt.')
            ms.append({
                'role': 'system',
                'content': m['text'] or '',
            })
        elif m['role'] == ChatMessageRole.HUMAN:
            ms.append({
                'role': 'user',
                'content': (await _make_openai_content(m)),
            })
        elif m['role'] == ChatMessageRole.AI:
            if m['media']:
                raise ValueError('This model does not support media in AI messages.')
            ms.append({
                'role': 'assistant',
                'content': m['text'],
            })
        else:
            raise ValueError(f"unknown message role: {m['role']}")
    def should_combine(
        m1: ChatCompletionMessageParam,
        m2: ChatCompletionMessageParam,
    ) -> Union[Literal[False], Tuple[Literal[True], ChatCompletionMessageParam]]:
        if m1['role'] == 'assistant' and m2['role'] == 'assistant':
            # This is the case where the model started with text and switched
            # to tool-calling part-way-through. We need to combine these
            # messages.
            assert m1.get('content') and not m1.get('tool_calls')
            assert not m2.get('content') and m2.get('tool_calls')
            m_combined: ChatCompletionMessageParam = {
                'role': 'assistant',
                'content': m1['content'],
                'tool_calls': m2['tool_calls'],
            }
            return True, m_combined
        return False
    return combine_pairs(ms, should_combine)


def _get_cost(
    payload: List[ChatCompletionChunk],
) -> Optional[Cost]:
    usages = [p.usage for p in payload if p.usage]
    if not usages:
        return None
    usage = usages[-1]
    return {
        'input_tokens': usage.prompt_tokens,
        'output_tokens': usage.completion_tokens,
        'total_tokens': usage.total_tokens,
        'cost_usd_cents': None,   # API doesn't give this, unfortunately
    }


def _build_messages_from_openai_payload(
    payload: List[ChatCompletionChunk],
    events: List[EventPayload],
) -> List[ChatMessage]:
    """
    We either have:
     - all AI events
     - all TOOL_CALL events
     - some AI events then it switches to TOOL_CALL events
    """
    cost = _get_cost(payload)
    raw = [p.to_dict() for p in payload]
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
        'text': ''.join([e[2] for e in ai_events if e[1] == 'text_event']),
        'media': None, # <-- the chat API doesn't know how to generate images (it only _reads_ images)
        'cost': cost,
        'raw': raw,
    } if len(ai_events) > 0 else None
    tool_message: Optional[ChatMessageToolCall] = {
        'role': ChatMessageRole.TOOL_CALL,
        'tools': [e[2] for e in tool_events if e[1] == 'tool_call_event'],
        'cost': cost,
        'raw': raw,
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


async def _handle_tools(
    messages: List[ChatMessage],
    tools_map: Dict[str, Callable],
) -> Union[List[ToolResult], None]:
    assert len(messages) > 0
    message = messages[-1]   # <-- the tool message will be last, if at all
    if message['role'] != ChatMessageRole.TOOL_CALL:
        return None
    to_gather: List[asyncio.Task[ToolResult]] = []
    for t in message['tools']:
        assert t['call_type'] == 'function'
        async def _go(t: ToolCall) -> ToolResult:
            call_id = 'unknown'
            try:
                call_id = t['call_id']
                func = tools_map[t['function']['name']]
                args = t['function']['arguments']
                if inspect.iscoroutinefunction(func):
                    res = await func(**json.loads(args))
                else:
                    def _wrapped_sync() -> Any:
                        return func(**json.loads(args))
                    loop = asyncio.get_running_loop()
                    res = await loop.run_in_executor(None, _wrapped_sync)
                return {'call_id': call_id, 'result': res}
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                return {'call_id': call_id, 'result': error}
        to_gather.append(asyncio.create_task(_go(t)))
    return await asyncio.gather(*to_gather)


def _build_tool_response_message(tool_results: List[ToolResult]) -> ChatMessage:
    return {
        'role': ChatMessageRole.TOOL_RES,
        'tools': tool_results,
        'cost': None,
        'raw': None,
    }


class LasagnaOpenAI(LLM):
    def __init__(self, model: str, **model_kwargs: Dict[str, Any]):
        known_model_names = [m['formal_name'] for m in OPENAI_KNOWN_MODELS]
        if model not in known_model_names:
            raise ValueError(f'unknown model: {model}')
        self.model = model
        self.model_kwargs = copy.deepcopy(model_kwargs or {})
        self.client = AsyncOpenAI()

    async def _run_once(
        self,
        event_callback: EventCallback,
        messages: List[ChatMessage],
        tools: List[Callable],
        force_tool: bool = False,
    ) -> List[ChatMessage]:
        tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven]
        if force_tool:
            if len(tools) != 1:
                raise ValueError(f"When `force_tool` is set, you must pass exactly one tool, not {len(tools)}.")
            tool_choice = {
                "type": "function",
                "function": {"name": tools[0].__name__},
            }
        else:
            tool_choice = NOT_GIVEN

        tools_spec = _convert_to_openai_tools(tools)

        frequency_penalty: Union[float, NotGiven] = cast(float, self.model_kwargs['frequency_penalty']) if 'frequency_penalty' in self.model_kwargs else NOT_GIVEN
        presence_penalty: Union[float, NotGiven] = cast(float, self.model_kwargs['presence_penalty']) if 'presence_penalty' in self.model_kwargs else NOT_GIVEN
        max_tokens: Union[int, NotGiven] = cast(int, self.model_kwargs['max_tokens']) if 'max_tokens' in self.model_kwargs else NOT_GIVEN
        stop: Union[List[str], NotGiven] = cast(List[str], self.model_kwargs['stop']) if 'stop' in self.model_kwargs else NOT_GIVEN
        temperature: Union[float, NotGiven] = cast(float, self.model_kwargs['temperature']) if 'temperature' in self.model_kwargs else NOT_GIVEN
        top_p: Union[float, NotGiven] = cast(float, self.model_kwargs['top_p']) if 'top_p' in self.model_kwargs else NOT_GIVEN
        user: Union[str, NotGiven] = cast(str, self.model_kwargs['user']) if 'user' in self.model_kwargs else NOT_GIVEN

        completion: AsyncIterator[ChatCompletionChunk] = await self.client.chat.completions.create(
            model        = self.model,
            messages     = (await _convert_to_openai_messages(messages)),
            tools        = tools_spec,
            tool_choice  = tool_choice,
            stream       = True,
            stream_options = {'include_usage': True},
            logprobs     = True,
            top_logprobs = 20,
            frequency_penalty = frequency_penalty,
            presence_penalty = presence_penalty,
            max_tokens = max_tokens,
            stop = stop,
            temperature = temperature,
            top_p = top_p,
            user = user,
        )

        raw_stream, rt_stream = adup(completion)

        events: List[EventPayload] = []

        async for event in _process_output_stream(_extract_deltas(rt_stream)):
            events.append(event)
            await event_callback(event)

        raw_payload = [v async for v in raw_stream]

        new_messages = _build_messages_from_openai_payload(raw_payload, events)

        return new_messages

    async def run(
        self,
        event_callback: EventCallback,
        messages: List[ChatMessage],
        tools: List[Callable],
        force_tool: bool = False,
        max_tool_iters: int = 5,
    ) -> List[ChatMessage]:
        messages = [*messages]  # shallow copy
        new_messages: List[ChatMessage] = []
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
            tools_results = await _handle_tools(new_messages_here, tools_map)
            if tools_results is None:
                break
            for tool_result in tools_results:
                await event_callback((ChatMessageRole.TOOL_RES, 'tool_res_event', tool_result))
            tool_response_message = _build_tool_response_message(tools_results)
            new_messages.append(tool_response_message)
            messages.append(tool_response_message)
        return new_messages
