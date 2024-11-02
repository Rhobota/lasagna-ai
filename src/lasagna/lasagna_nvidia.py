"""
This module is the Lasagna adapter for the NVIDIA NIM/NGC service.

For more information about the NVIDIA services this adapter is for, see:
 - https://org.ngc.nvidia.com/
 - https://build.nvidia.com/explore/reasoning
 - https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html
"""

from .types import (
    ModelSpec,
    Message,
    MessageContent,
    MessageToolCall,
    EventCallback,
    EventPayload,
    Model,
    ToolCall,
    Cost,
    ExtractionType,
)

from .stream import (
    apeek,
    adup,
    prefix_stream,
)

from .util import (
    combine_pairs,
    convert_to_image_url,
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

from .known_models import NVIDIA_KNOWN_MODELS

from openai import AsyncOpenAI, NOT_GIVEN, NotGiven, pydantic_function_tool
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionMessageParam,
    ChatCompletionContentPartParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai import APIError

from typing import (
    List, Callable, Type, AsyncIterator, Any,
    Tuple, Dict, Optional, Union, Literal,
    cast,
)

import asyncio
import copy
import json
import os

import logging

_LOG = logging.getLogger(__name__)


async def _process_text_stream(
    stream: AsyncIterator[Tuple[ChoiceDelta, Union[str, None]]],
) -> AsyncIterator[EventPayload]:
    async for record in stream:
        delta, finish_reason = record
        text = delta.content
        if finish_reason is not None:
            if text is not None:
                yield 'ai', 'text_event', str(text)
            return
        elif delta.tool_calls is not None:
            # The model is switching from text to tools!
            yield 'ai', 'text_event', "\n\n"
            fixed_stream = prefix_stream([record], stream)
            substream = _process_tool_call_stream(fixed_stream)
            async for subval in substream:
                yield subval
            return
        elif text is not None:
            yield 'ai', 'text_event', str(text)


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
                yield 'tool_call', 'text_event', ")\n"   # <-- again, assumes no index-interleave
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
                yield 'tool_call', 'text_event', f"{n}("   # <-- assumes no index-interleave
            else:
                # assumes nothing but the argument is in the delta message...
                args = args_by_index[index]
                assert tc.function
                assert tc.function.arguments
                assert not tc.function.name
                a_delta: str = tc.function.arguments
                args.append(a_delta)
                yield 'tool_call', 'text_event', a_delta
    if last_index is not None:
        yield 'tool_call', 'text_event', ")"   # <-- again, assumes no index-interleave
    for index in sorted(recs_by_index.keys()):
        rec = recs_by_index[index]
        rec['function']['arguments'] = ''.join(args_by_index[index])
        yield 'tool_call', 'tool_call_event', rec


async def _process_output_stream(
    stream: AsyncIterator[Tuple[ChoiceDelta, Union[str, None]]],
) -> AsyncIterator[EventPayload]:
    first, stream = await apeek(stream, n=1)
    first_delta, _ = first[0]
    is_text = first_delta.tool_calls is None   # <-- hacky, but works?
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


def _convert_to_openai_tool(tool: Callable) -> ChatCompletionToolParam:
    description, params = get_tool_params(tool)
    return {
        'type': 'function',
        'function': {
            'name': get_name(tool),
            'description': description,
            'parameters': convert_to_json_schema(params),
        },
    }


def _convert_to_openai_tools(tools: List[Callable]) -> Union[NotGiven, List[ChatCompletionToolParam]]:
    if len(tools) == 0:
        return NOT_GIVEN
    specs = [_convert_to_openai_tool(tool) for tool in tools]
    for spec in specs:
        if 'parameters' in spec['function']:
            p = spec['function']['parameters']
            if 'additionalProperties' in p:
                del p['additionalProperties']
    return specs


async def _make_openai_content(
    message: MessageContent,
) -> Union[str, List[ChatCompletionContentPartParam]]:
    ret: List[ChatCompletionContentPartParam] = []
    if message['text'] and not message.get('media'):
        return message['text']
    elif message['text']:
        ret.append({
            'type': 'text',
            'text': message['text'],
        })
    if 'media' in message:
        for m in message['media']:
            if m['type'] == 'image':
                ret.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': (await convert_to_image_url(m['image'])),
                    },
                })
            else:
                raise ValueError(f"unknown media type: {m['type']}")
    if len(ret) == 0:
        raise ValueError('no content in this message!')
    return ret


async def _convert_to_openai_messages(messages: List[Message]) -> List[ChatCompletionMessageParam]:
    ms: List[ChatCompletionMessageParam] = []
    for m in messages:
        if m['role'] == 'tool_call':
            tool_calls = m['tools']
            for tool_call in tool_calls:
                assert tool_call['call_type'] == 'function'
            ms.append({
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': tool_call['call_id'],
                        'type': tool_call['call_type'],
                        'function': {
                            'name': tool_call['function']['name'],
                            'arguments': tool_call['function']['arguments'],
                        },
                    }
                    for tool_call in tool_calls
                ],
            })
        elif m['role'] == 'tool_res':
            tool_results = m['tools']
            for tool_result in tool_results:
                content = extract_tool_result_as_sting(tool_result)
                ms.append({
                    'role': 'tool',
                    'content': content,
                    'tool_call_id': tool_result['call_id'],
                })
        elif m['role'] == 'system':
            if 'media' in m and len(m['media']) > 0:
                raise ValueError('This model does not support media in the system prompt.')
            ms.append({
                'role': 'system',
                'content': m['text'] or '',
            })
        elif m['role'] == 'human':
            ms.append({
                'role': 'user',
                'content': (await _make_openai_content(m)),
            })
        elif m['role'] == 'ai':
            if 'media' in m and len(m['media']) > 0:
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
            assert ('content' in m1 and m1['content']) and ('tool_calls' not in m1 or not m1['tool_calls'])
            assert ('content' not in m2 or not m2['content']) and ('tool_calls' in m2 and m2['tool_calls'])
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
    }


def _build_messages_from_openai_payload(
    payload: List[ChatCompletionChunk],
    events: List[EventPayload],
) -> List[Message]:
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
        if event[0] == 'ai'
    ]
    tool_events = [
        event
        for event in events
        if event[0] == 'tool_call'
    ]
    ai_message: Optional[Message] = {
        'role': 'ai',
        'text': ''.join([e[2] for e in ai_events if e[1] == 'text_event']),
        'cost': cost,
        'raw': raw,
    } if len(ai_events) > 0 else None
    tool_message: Optional[MessageToolCall] = {
        'role': 'tool_call',
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


def _log_dumps(val: Any) -> str:
    if isinstance(val, dict):
        return json.dumps(val)
    else:
        return str(val)


class LasagnaNVIDIA(Model):
    def __init__(self, model: str, **model_kwargs: Dict[str, Any]):
        known_model_names = [m['formal_name'] for m in NVIDIA_KNOWN_MODELS]
        if model not in known_model_names:
            _LOG.warning(f'untested model: {model} (may or may not work)')
        self.model = model
        self.model_kwargs = copy.deepcopy(model_kwargs or {})
        self.n_retries: int = cast(int, self.model_kwargs['retries']) if 'retries' in self.model_kwargs else 3
        if not isinstance(self.n_retries, int) or self.n_retries < 0:
            raise ValueError(f"model_kwargs['retries'] must be a non-negative integer (got {self.model_kwargs['retries']})")
        self.model_spec: ModelSpec = {
            'provider': 'nvidia',
            'model': self.model,
            'model_kwargs': self.model_kwargs,
        }

    def config_hash(self) -> str:
        return recursive_hash(None, {
            'provider': 'nvidia',
            'model': self.model,
            'model_kwargs': self.model_kwargs,
        })

    def _make_client(self) -> AsyncOpenAI:
        default_api_key = os.environ.get('NGC_API_KEY', None)
        default_base_url = 'https://integrate.api.nvidia.com/v1'
        api_key: Union[str, None] = cast(str, self.model_kwargs['api_key']) if 'api_key' in self.model_kwargs else default_api_key
        base_url: Union[str, None] = cast(str, self.model_kwargs['base_url']) if 'base_url' in self.model_kwargs else default_base_url
        client = AsyncOpenAI(
            api_key  = api_key,
            base_url = base_url,
        )
        return client

    async def _run_once(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools_spec: Union[NotGiven, List[ChatCompletionToolParam]],
        force_tool: bool,
        parallel_tool_calls: Union[NotGiven, bool],
    ) -> List[Message]:
        tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven]
        if force_tool:
            if not tools_spec or len(tools_spec) == 0:
                raise ValueError(f"When `force_tool` is set, you must pass at least one tool!")
            elif len(tools_spec) == 1:
                tool_choice = {
                    "type": "function",
                    "function": {"name": tools_spec[0]['function']['name']},
                }
            else:
                tool_choice = 'required'  # <-- model must use a tool, but is allowed to choose which one on its own
        else:
            tool_choice = NOT_GIVEN  # <-- if tools given, the model can choose to use them or not

        openai_messages = await _convert_to_openai_messages(messages)

        logprobs: Union[bool, NotGiven] = cast(bool, self.model_kwargs['logprobs']) if 'logprobs' in self.model_kwargs else NOT_GIVEN
        top_logprobs: Union[int, NotGiven] = cast(int, self.model_kwargs['top_logprobs']) if 'top_logprobs' in self.model_kwargs else (5 if logprobs is True else NOT_GIVEN)
        frequency_penalty: Union[float, NotGiven] = cast(float, self.model_kwargs['frequency_penalty']) if 'frequency_penalty' in self.model_kwargs else NOT_GIVEN
        presence_penalty: Union[float, NotGiven] = cast(float, self.model_kwargs['presence_penalty']) if 'presence_penalty' in self.model_kwargs else NOT_GIVEN
        max_tokens: Union[int, NotGiven] = cast(int, self.model_kwargs['max_tokens']) if 'max_tokens' in self.model_kwargs else 1024
        stop: Union[List[str], NotGiven] = cast(List[str], self.model_kwargs['stop']) if 'stop' in self.model_kwargs else NOT_GIVEN
        temperature: Union[float, NotGiven] = cast(float, self.model_kwargs['temperature']) if 'temperature' in self.model_kwargs else NOT_GIVEN
        top_p: Union[float, NotGiven] = cast(float, self.model_kwargs['top_p']) if 'top_p' in self.model_kwargs else NOT_GIVEN
        user: Union[str, NotGiven] = cast(str, self.model_kwargs['user']) if 'user' in self.model_kwargs else NOT_GIVEN

        _LOG.info(f"Invoking {self.model} with:\n  messages: {_log_dumps(openai_messages)}\n  tools: {_log_dumps(tools_spec)}\n  tool_choice: {tool_choice}")

        client = self._make_client()
        completion: AsyncIterator[ChatCompletionChunk] = await client.chat.completions.create(
            model        = self.model,
            messages     = openai_messages,
            tools        = tools_spec,
            tool_choice  = tool_choice,
            parallel_tool_calls = parallel_tool_calls,
            stream       = True,
            #stream_options = {'include_usage': True},
            logprobs     = logprobs,
            top_logprobs = top_logprobs,
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

        _LOG.info(f"Finished {self.model} with usage: {_log_dumps(new_messages[-1].get('cost'))}")

        return new_messages

    async def _retrying_run_once(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools_spec: Union[NotGiven, List[ChatCompletionToolParam]],
        force_tool: bool,
        parallel_tool_calls: Union[NotGiven, bool],
    ) -> List[Message]:
        last_error: Union[APIError, None] = None
        assert self.n_retries + 1 > 0   # <-- we know this is true from the check in __init__
        for delay_on_error in exponential_backoff_retry_delays(self.n_retries + 1):
            try:
                await event_callback(('transaction', 'start', ('nvidia', self.model)))
                try:
                    new_messages = await self._run_once(
                        event_callback = event_callback,
                        messages = messages,
                        tools_spec = tools_spec,
                        force_tool = force_tool,
                        parallel_tool_calls = parallel_tool_calls,
                    )
                except:
                    await event_callback(('transaction', 'rollback', None))
                    raise
                await event_callback(('transaction', 'commit', None))
                return new_messages
            except APIError as e:
                # Some errors should be retried, some should not. Below
                # is the logic to decide when to retry vs when to not.
                # It's likely this will change as we get more usage and see
                # where NVIDIA tends to fail, and when we know more what is
                # recoverable vs not.
                last_error = e
                if e.type == 'invalid_request_error':
                    if e.code == 'context_length_exceeded':
                        raise
                    elif e.code == 'invalid_api_key':
                        raise
                    else:
                        raise
                elif e.type == 'server_error':
                    pass  # <-- we will retry this one! I've seen these work when you just try again.
                else:
                    pass  # <-- this must be one we don't know about yet, so ... recoverable, maybe?
                if delay_on_error > 0.0:
                    _LOG.warning(f"Got a maybe-recoverable error (will retry in {delay_on_error:.2f} seconds) for model `{self.model}`: {e}")
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
        tools_spec = _convert_to_openai_tools(tools)
        tools_map = {get_name(tool): tool for tool in tools}
        for _ in range(max_tool_iters):
            new_messages_here = await self._retrying_run_once(
                event_callback = event_callback,
                messages       = messages,
                tools_spec     = tools_spec,
                force_tool     = force_tool,
                parallel_tool_calls = NOT_GIVEN,
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
        tools_spec = [pydantic_function_tool(ensure_pydantic_model(extraction_type))]

        new_messages = await self._retrying_run_once(
            event_callback = event_callback,
            messages       = messages,
            tools_spec     = tools_spec,
            force_tool     = True,
            parallel_tool_calls = False,
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
