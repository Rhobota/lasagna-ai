from .types import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageRole,
    EventCallback,
    EventPayload,
    LLM,
    ToolCall,
    ToolParam,
    ToolResult,
    ModelRecord,
)

from .stream import (
    apeek,
    adup,
    prefix_stream,
)

from .util import parse_docstring

from .registrar import register_model_provider

from openai import AsyncOpenAI, NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from typing import (
    List, Callable, AsyncIterator, Any,
    Tuple, Dict, Optional, Union,
)

import copy
import json


_KNOWN_MODELS: List[ModelRecord] = [
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
                yield ChatMessageRole.AI, 'text', str(text)
            return
        if text is None:
            # The model is switching from text to tools!
            yield ChatMessageRole.AI, 'text', "\n\n"
            put_back_val: Tuple[ChoiceDelta, Union[str, None]] = (delta, finish_reason)
            fixed_stream = prefix_stream([put_back_val], stream)
            substream = _process_tool_call_stream(fixed_stream)
            async for subval in substream:
                yield subval
            return
        yield ChatMessageRole.AI, 'text', str(text)


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
                yield ChatMessageRole.TOOL_CALL, 'text', ")\n"   # <-- again, assumes no index-interleave
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
                yield ChatMessageRole.TOOL_CALL, 'text', f"{n}("   # <-- assumes no index-interleave
            else:
                # assumes nothing but the argument is in the delta message...
                args = args_by_index[index]
                assert tc.function
                assert tc.function.arguments
                assert not tc.function.name
                a_delta: str = tc.function.arguments
                args.append(a_delta)
                yield ChatMessageRole.TOOL_CALL, 'text', a_delta
    if last_index is not None:
        yield ChatMessageRole.TOOL_CALL, 'text', ")"   # <-- again, assumes no index-interleave
    for index in sorted(recs_by_index.keys()):
        rec = recs_by_index[index]
        rec['function']['arguments'] = ''.join(args_by_index[index])
        yield ChatMessageRole.TOOL_CALL, 'tool_call', rec


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
    events: List[EventPayload],
) -> List[ChatMessage]:
    """
    We either have:
     - all AI events
     - all TOOL_CALL events
     - some AI events then it switches to TOOL_CALL events
    """
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
        'text': ''.join([e[2] for e in ai_events if e[1] == 'text']),
        'cost': None,  # TODO: OpenAI's API doesn't return this info in streaming mode! Hopefully they will in the future.
        'raw': raw,
    } if len(ai_events) > 0 else None
    tool_message: Optional[ChatMessageToolCall] = {
        'role': ChatMessageRole.TOOL_CALL,
        'tools': [e[2] for e in tool_events if e[1] == 'tool_call'],
        'cost': None,  # TODO: OpenAI's API doesn't return this info in streaming mode! Hopefully they will in the future.
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


def _handle_tools(
    messages: List[ChatMessage],
    tools_map: Dict[str, Callable],
) -> Union[List[ToolResult], None]:
    # TODO: this needs to be async and delegate to a threadpool behind the scenes
    assert len(messages) > 0
    message = messages[-1]   # <-- the tool message will be last, if at all
    if message['role'] != ChatMessageRole.TOOL_CALL:
        return None
    results: List[ToolResult] = []
    for t in message['tools']:
        assert t['call_type'] == 'function'
        call_id = 'unknown'
        try:
            call_id = t['call_id']
            func = tools_map[t['function']['name']]
            args = t['function']['arguments']
            res = func(**json.loads(args))
            results.append({'call_id': call_id, 'result': res})
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            results.append({'call_id': call_id, 'result': error})
    return results


def _build_tool_response_message(tool_results: List[ToolResult]) -> ChatMessage:
    return {
        'role': ChatMessageRole.TOOL_RES,
        'tools': tool_results,
        'cost': None,
        'raw': None,
    }


class LasagnaOpenAI(LLM):
    def __init__(self, model: str, **model_kwargs: Dict[str, Any]):
        known_model_names = [m['formal_name'] for m in _KNOWN_MODELS]
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
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven
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

        completion: AsyncIterator[ChatCompletionChunk] = await self.client.chat.completions.create(
            model        = self.model,
            messages     = _convert_to_openai_messages(messages),
            tools        = tools_spec,
            tool_choice  = tool_choice,
            stream       = True,
            logprobs     = True,
            top_logprobs = 20,
            #**self.model_kwargs,    # TODO
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
            tools_results = _handle_tools(new_messages_here, tools_map)
            if tools_results is None:
                break
            for tool_result in tools_results:
                await event_callback((ChatMessageRole.TOOL_RES, 'tool_res', tool_result))
            tool_response_message = _build_tool_response_message(tools_results)
            new_messages.append(tool_response_message)
            messages.append(tool_response_message)
        return new_messages


register_model_provider(
    key  = 'openai',
    name = 'OpenAI',
    factory = LasagnaOpenAI,
    models = _KNOWN_MODELS,
)