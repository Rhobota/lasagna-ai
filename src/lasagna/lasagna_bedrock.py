"""
This module is the Lasagna adapter for AWS Bedrock.

For more information about the AWS Bedrock, see:
 - https://docs.aws.amazon.com/bedrock/
"""

from .types import (
    MessageContent,
    MessageToolCall,
    MessageToolResult,
    ModelSpec,
    Message,
    EventCallback,
    Model,
    ExtractionType,
    Cost,
    ToolCall,
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

from .pydantic_util import build_and_validate, ensure_pydantic_model

from .known_models import BEDROCK_KNOWN_MODELS

from typing import (
    List, Callable, Type, AsyncIterable, Any,
    Tuple, Dict, Union,
)

import boto3.session  # type: ignore
from botocore.exceptions import ClientError  # type: ignore

from openai.lib._pydantic import to_strict_json_schema

import os
import asyncio
import copy
import json

import logging

_LOG = logging.getLogger(__name__)


def _convert_to_bedrock_tool(tool: Callable) -> Dict:
    description, params = get_tool_params(tool)
    schema = convert_to_json_schema(params)
    if 'additionalProperties' in schema:
        del schema['additionalProperties']
    return {
        'toolSpec': {
            'name': get_name(tool),
            'description': description,
            'inputSchema': {
                'json': schema,
            },
        },
    }


def _convert_to_bedrock_tools(tools: List[Callable]) -> Union[None, List[Dict]]:
    if len(tools) == 0:
        return None
    specs = [_convert_to_bedrock_tool(tool) for tool in tools]
    return specs


async def _build_bedrock_content(message: MessageContent) -> List[Dict]:
    r: List[Dict] = []
    if message['text']:
        r.append({
            'text': message['text'],
        })
    if 'media' in message:
        for m in message['media']:
            assert m['type'] == 'image', 'we only support images, for now'
            mimetype, data = await convert_to_image_base64(m['image'])
            assert mimetype.startswith('image/')
            mimetype = mimetype.split('/')[1]
            assert mimetype in ['png', 'jpeg', 'gif', 'webp']
            r.append({
                'image': {
                    'format': mimetype,
                    'source': {
                        'bytes': data,
                    },
                },
            })
    return r


async def _build_bedrock_tool_use(message: MessageToolCall) -> List[Dict]:
    ret: List[Dict] = []
    for tool in message['tools']:
        if tool['call_type'] == 'function':
            ret.append({
                'toolUse': {
                    'toolUseId': tool['call_id'],
                    'name': tool['function']['name'],
                    'input': json.loads(tool['function']['arguments']),
                },
            })
        else:
            raise ValueError(f"unknown tool type: {tool['call_type']}")
    if len(ret) == 0:
        raise ValueError(f"no content")
    return ret


async def _build_bedrock_tool_result(message: MessageToolResult) -> List[Dict]:
    ret: List[Dict] = []
    for tool_result in message['tools']:
        content = extract_tool_result_as_sting(tool_result)
        is_error = 'is_error' in tool_result and tool_result['is_error']
        ret.append({
            'toolResult': {
                'toolUseId': tool_result['call_id'],
                'content': [
                    {
                        'text': content,
                    },
                ],
                'status': 'error' if is_error else 'success',
            },
        })
    if len(ret) == 0:
        raise ValueError(f"no content")
    return ret


def _collapse_bedrock_messages(messages: List[Dict]) -> List[Dict]:
    ms: List[Dict] = []
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


async def _convert_to_bedrock_messages(
    messages: List[Message],
) -> Tuple[Union[None, str], List[Dict]]:
    system_prompt: Union[str, None] = None
    ret: List[Dict] = []
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
                    'content': await _build_bedrock_content(m),
                })
            elif m['role'] == 'human':
                ret.append({
                    'role': 'user',
                    'content': await _build_bedrock_content(m),
                })
            elif m['role'] == 'tool_call':
                ret.append({
                    'role': 'assistant',
                    'content': await _build_bedrock_tool_use(m),
                })
            elif m['role'] == 'tool_res':
                ret.append({
                    'role': 'user',
                    'content': await _build_bedrock_tool_result(m),
                })
            else:
                raise ValueError(f"Unknown role: {m['role']}")
    ret = [r for r in ret if len(list(r['content'])) > 0]
    ret = _collapse_bedrock_messages(ret)
    return system_prompt, ret


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


async def _process_bedrock_stream(
    stream: AsyncIterable[Dict],
    event_callback: EventCallback,
) -> List[Message]:
    raw_events: List[Dict] = []
    partial_messages: List[Dict] = []

    def _init_message(role: str) -> None:
        cost: Cost = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
        }
        partial_messages.append({
            'role': role,
            'blocks': {},
            'cost': cost,
        })

    def _get_block(curr_message: Dict, this_block_index: int) -> Dict:
        if this_block_index not in curr_message['blocks']:
            curr_message['blocks'][this_block_index] = {
                'type': 'text',  # <-- default to text unless proven otherwise later
                'content': [],
            }
        this_block = curr_message['blocks'][this_block_index]
        assert isinstance(this_block, dict)
        return this_block

    async for event in stream:
        raw_events.append(event)

        # Frustratingly, Bedrock sometimes sends a `contentBlockDelta`
        # without first sending `messageStart` nor `contentBlockStart`.
        # So, we have to be ready for that!

        if 'messageStart' in event:
            _init_message(event['messageStart']['role'])

        elif 'messageStop' in event:
            pass   # nothing to do here

        elif 'contentBlockStart' in event:
            payload = event['contentBlockStart']
            if len(partial_messages) == 0:
                _init_message('assistant')
            curr_message = partial_messages[-1]
            this_block = _get_block(curr_message, payload['contentBlockIndex'])

            if 'toolUse' in payload['start']:
                tool_use = payload['start']['toolUse']
                call_id = tool_use['toolUseId']
                func_name = tool_use['name']
                this_block['type'] = 'tool_use'
                text = f'{func_name}('
                await event_callback(('tool_call', 'text_event', text))
                this_block['call_id'] = call_id
                this_block['func_name'] = func_name
            else:
                _LOG.warning(f'unknown bedrock `start` event: {event}')

        elif 'contentBlockDelta' in event:
            payload = event['contentBlockDelta']
            if len(partial_messages) == 0:
                _init_message('assistant')
            curr_message = partial_messages[-1]
            this_block = _get_block(curr_message, payload['contentBlockIndex'])

            delta = payload['delta']
            if 'text' in delta:
                assert this_block['type'] == 'text'
                text = delta['text']
                await event_callback(('ai', 'text_event', text))
                this_block['content'].append(text)
            elif 'toolUse' in delta:
                assert this_block['type'] == 'tool_use'
                text = delta['toolUse']['input']
                await event_callback(('tool_call', 'text_event', text))
                this_block['content'].append(text)
            else:
                _LOG.warning(f'unknown bedrock `delta` type: {event}')

        elif 'contentBlockStop' in event:
            payload = event['contentBlockStop']
            if len(partial_messages) == 0:
                _init_message('assistant')
            curr_message = partial_messages[-1]
            this_block_index = payload['contentBlockIndex']
            assert this_block_index in curr_message['blocks']
            this_block = curr_message['blocks'][this_block_index]

            if this_block['type'] == 'tool_use':
                text = ')\n'
                await event_callback(('tool_call', 'text_event', text))
                tool_call: ToolCall = {
                    'call_type': 'function',
                    'call_id': this_block['call_id'],
                    'function': {
                        'name': this_block['func_name'],
                        'arguments': ''.join(this_block['content']),
                    },
                }
                await event_callback(('tool_call', 'tool_call_event', tool_call))
                this_block['did_emit'] = True

        elif 'metadata' in event:
            payload = event['metadata']
            if len(partial_messages) == 0:
                _init_message('assistant')
            curr_message = partial_messages[-1]

            usage = payload['usage']
            curr_message['cost']['input_tokens'] += usage['inputTokens']
            curr_message['cost']['output_tokens'] += usage['outputTokens']
            curr_message['cost']['total_tokens'] += usage['totalTokens']

        else:
            _LOG.warning(f'unknown bedrock streaming event: {event}')

    sorted_blocks = []
    for pm in partial_messages:
        assert pm['role'] == 'assistant'
        sorted_blocks.extend([
            pm['blocks'][idx]
            for idx in sorted(pm['blocks'].keys())
        ])

    ms: List[Message] = []
    for block in sorted_blocks:
        if block['type'] == 'text':
            ms.append({
                'role': 'ai',
                'text': ''.join(block['content']),
            })
        elif block['type'] == 'tool_use':
            tool_call_here: ToolCall = {
                'call_type': 'function',
                'call_id': block['call_id'],
                'function': {
                    'name': block['func_name'],
                    'arguments': ''.join(block['content']),
                },
            }
            ms.append({
                'role': 'tool_call',
                'tools': [tool_call_here],
            })
            if not block.get('did_emit', False):
                text = ')\n'
                await event_callback(('tool_call', 'text_event', text))
                await event_callback(('tool_call', 'tool_call_event', tool_call_here))
        else:
            raise ValueError(f"unknown content type: {block['type']}")
    ms = _collapse_tool_call_messages(ms)
    if len(ms) == 0:
        ms.append({
            'role': 'ai',
            'text': None,  # <-- sometimes it generates nothing, so we'll show that
        })

    total_cost: Cost = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
    }
    for pm in partial_messages:
        total_cost['input_tokens'] += pm['cost']['input_tokens']
        total_cost['output_tokens'] += pm['cost']['output_tokens']
        total_cost['total_tokens'] += pm['cost']['total_tokens']

    last_message = ms[-1]
    last_message['cost'] = total_cost
    last_message['raw'] = {
        'events': raw_events,
    }

    return ms


def _log_dumps(val: Any) -> str:
    if isinstance(val, dict):
        return json.dumps(val)
    else:
        return str(val)


class LasagnaBedrock(Model):
    def __init__(self, model: str, **model_kwargs: Any):
        known_model_names = [m['formal_name'] for m in BEDROCK_KNOWN_MODELS]
        if model not in known_model_names:
            _LOG.warning(f'untested model: {model} (may or may not work)')
        self.model = model
        self.model_kwargs = copy.deepcopy(model_kwargs or {})
        self.n_retries: int = int(self.model_kwargs['retries']) if 'retries' in self.model_kwargs else 3
        if not isinstance(self.n_retries, int) or self.n_retries < 0:
            raise ValueError(f"model_kwargs['retries'] must be a non-negative integer (got {self.model_kwargs['retries']})")
        self.model_spec: ModelSpec = {
            'provider': 'bedrock',
            'model': self.model,
            'model_kwargs': self.model_kwargs,
        }

        # The boto3 client!
        self.session = boto3.session.Session(
            aws_access_key_id     = self.model_kwargs.get('aws_access_key_id',     os.environ.get('AWS_ACCESS_KEY_ID')),
            aws_secret_access_key = self.model_kwargs.get('aws_secret_access_key', os.environ.get('AWS_SECRET_ACCESS_KEY')),
            aws_session_token     = self.model_kwargs.get('aws_session_token',     os.environ.get('AWS_SESSION_TOKEN')),
            region_name           = self.model_kwargs.get('region_name',           os.environ.get('AWS_REGION')),
            profile_name          = self.model_kwargs.get('profile_name',          os.environ.get('AWS_PROFILE')),
        )
        self.bedrock_client = self.session.client(service_name='bedrock-runtime')
        # Note: `bedrock_client` is thread-safe, so you *can* safely pass
        #       it around to the event loop's thread pool.
        # See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/clients.html#multithreading-or-multiprocessing-with-clients

    def config_hash(self) -> str:
        return recursive_hash(None, {
            'provider': 'bedrock',
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
        system_prompt, bedrock_messages = await _convert_to_bedrock_messages(messages)

        system = [{'text' : system_prompt}] if system_prompt is not None else None

        tool_config: Union[None, Dict] = None
        if tools_spec:
            tool_config = {
                'tools': tools_spec,
            }
            if force_tool:
                if len(tools_spec) == 1:
                    name = tools_spec[0]['toolSpec']['name']
                    tool_config['toolChoice'] = {'tool' : {'name' : name}}
                else:
                    tool_config['toolChoice'] = {'any': {}}
            else:
                tool_config['toolChoice'] = {'auto' : {}}
        elif force_tool:
            raise ValueError(f"When `force_tool` is set, you must pass at least one tool!")

        max_tokens: int = int(self.model_kwargs['max_tokens']) if 'max_tokens' in self.model_kwargs else 4096
        stop: Union[List[str], None] = self.model_kwargs['stop'] if 'stop' in self.model_kwargs else None
        temperature: Union[float, None] = float(self.model_kwargs['temperature']) if 'temperature' in self.model_kwargs else None
        top_p: Union[float, None] = float(self.model_kwargs['top_p']) if 'top_p' in self.model_kwargs else None

        inference_config: Dict[str, Union[int, float, str, List[str]]] = {}
        inference_config['maxTokens'] = max_tokens
        if stop is not None:
            assert isinstance(stop, list)
            for s in stop:
                assert isinstance(s, str)
            inference_config['stopSequences'] = stop
        if temperature is not None:
            inference_config['temperature'] = temperature
        if top_p is not None:
            inference_config['topP'] = top_p

        _LOG.info(f"Invoking {self.model} with:\n  system_prompt: {system_prompt}\n  messages: {_log_dumps(bedrock_messages)}\n  tools: {_log_dumps(tools_spec)}\n  force_tool: {force_tool}")

        params = {
            'modelId': self.model,
            'messages': bedrock_messages,
            'inferenceConfig': inference_config,
        }

        if system is not None:
            params['system'] = system

        if tool_config is not None:
            params['toolConfig'] = tool_config

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Union[Dict, None, Exception]] = asyncio.Queue()

        def _run_bedrock_model() -> None:
            try:
                response = self.bedrock_client.converse_stream(**params)
                stream = response.get('stream')
                assert stream, 'Expected to get a stream!'
                for event in stream:
                    assert isinstance(event, dict)
                    loop.call_soon_threadsafe(queue.put_nowait, event)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)

        task = loop.run_in_executor(None, _run_bedrock_model)

        async def _generator() -> AsyncIterable[Dict]:
            while True:
                event = await queue.get()
                if event is None:
                    break
                elif isinstance(event, Exception):
                    raise event
                yield event

        new_messages = await _process_bedrock_stream(_generator(), event_callback)

        await task

        _LOG.info(f"Finished {self.model} with usage: {_log_dumps(new_messages[-1].get('cost'))}")

        return new_messages

    async def _retrying_run_once(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools_spec: Union[None, List[Dict]],
        force_tool: bool,
    ) -> List[Message]:
        last_error: Union[ClientError, None] = None
        assert self.n_retries + 1 > 0   # <-- we know this is true from the check in __init__
        for delay_on_error in exponential_backoff_retry_delays(self.n_retries + 1):
            try:
                await event_callback(('transaction', 'start', ('bedrock', self.model)))
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
            except ClientError as e:
                # Some errors should be retried, some should not. Below
                # is the logic to decide when to retry vs when to not.
                # It's likely this will change as we get more usage and see
                # where Bedrock tends to fail, and when we know more what is
                # recoverable vs not.
                last_error = e
                status = e.response['ResponseMetadata']['HTTPStatusCode']
                if status in [403, 404, 400]:
                    raise   # not recoverable!
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
        tools_spec = _convert_to_bedrock_tools(tools)
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
                'toolSpec': {
                    'name': get_name(extraction_type),
                    #'description': ... filled below ...
                    'inputSchema': {
                        'json': to_strict_json_schema(ensure_pydantic_model(extraction_type)),
                    },
                },
            },
        ]

        docstr = getattr(extraction_type, '__doc__', None)
        if docstr:
            tools_spec[0]['toolSpec']['description'] = docstr

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
