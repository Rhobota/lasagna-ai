import pytest

import os
import tempfile

from typing import List, Dict, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel

from lasagna.types import EventPayload, Message, ToolCall

from lasagna.stream import fake_async

from lasagna.lasagna_ollama import (
    _convert_to_ollama_tool,
    _convert_to_ollama_tools,
    _convert_to_ollama_media,
    _convert_to_ollama_tool_calls,
    _convert_to_ollama_messages,
    _set_cost_raw,
    _process_stream,
    _wrap_event_callback_convert_ai_text_to_tool_call_text,
    _get_ollama_format_for_structured_output,
    LasagnaOllama,
)


def sample_tool(a: int, b: str = 'hi') -> str:
    """
    A sample tool.
    :param: a: int: the value of a
    :param: b: str: (optional) the value of b
    """
    return b * a


_sample_tool_correct_schema = {
    'type': 'function',
    'function': {
        'name': 'sample_tool',
        'description': 'A sample tool.',
        'parameters': {
            'type': 'object',
            'properties': {
                'a': {
                    'type': 'integer',
                    'description': 'the value of a',
                },
                'b': {
                    'type': 'string',
                    'description': '(optional) the value of b',
                },
            },
            'required': ['a'],
            'additionalProperties': False,
        },
    },
}


_sample_events_streaming_text: List[Dict] = [
    {'message': {'role': 'assistant', 'content': 'Hello'}, 'done': False},
    {'message': {'role': 'assistant', 'content': '!'}, 'done': False},
    {'message': {'role': 'assistant', 'content': ' How'}, 'done': False},
    {'message': {'role': 'assistant', 'content': ' can'}, 'done': False},
    {'message': {'role': 'assistant', 'content': ' I'}, 'done': False},
    {'message': {'role': 'assistant', 'content': ' assist'}, 'done': False},
    {'message': {'role': 'assistant', 'content': ' you'}, 'done': False},
    {'message': {'role': 'assistant', 'content': ' today'}, 'done': False},
    {'message': {'role': 'assistant', 'content': '?'}, 'done': False},
    {'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 290565553, 'load_duration': 4737247, 'prompt_eval_count': 20, 'prompt_eval_duration': 27000000, 'eval_count': 10, 'eval_duration': 257000000},
]


_sample_events_tool_call: List[Dict] = [
    # Ollama cannot stream in this case, so we get a single message here:
    {'message': {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'evaluate_math_expression', 'arguments': {'expression': '2.5 * sin(4.5)'}}}]}, 'done_reason': 'stop', 'done': True, 'total_duration': 799653463, 'load_duration': 4977833, 'prompt_eval_count': 100, 'prompt_eval_duration': 27000000, 'eval_count': 29, 'eval_duration': 765000000},
]


_sample_events_structued_output: List[Dict] = [
    # Ollama cannot stream in this case, so we get a single message here:
    {'message': {'role': 'assistant', 'content': '{"base":2,"exponent":101}'}, 'done_reason': 'stop', 'done': True, 'total_duration': 426267794, 'load_duration': 4418431, 'prompt_eval_count': 28, 'prompt_eval_duration': 21000000, 'eval_count': 15, 'eval_duration': 399000000},
]


def test_convert_to_ollama_tool():
    assert _convert_to_ollama_tool(sample_tool) == _sample_tool_correct_schema


def test_convert_to_ollama_tools():
    assert _convert_to_ollama_tools([]) is None
    assert _convert_to_ollama_tools([sample_tool]) == [
        _sample_tool_correct_schema,
    ]


@pytest.mark.asyncio
async def test_convert_to_ollama_media():
    with tempfile.TemporaryDirectory() as tmp:
        fn1 = os.path.join(tmp, 'a.png')
        fn2 = os.path.join(tmp, 'b.png')
        with open(fn1, 'wb') as f:
            f.write(b'1234')
        with open(fn2, 'wb') as f:
            f.write(b'1235')
        assert await _convert_to_ollama_media([
            {
                'type': 'image',
                'image': fn1,
            },
            {
                'type': 'image',
                'image': fn2,
            },
        ]) == {
            'images': ['MTIzNA==', 'MTIzNQ=='],
        }


def test_convert_to_ollama_tool_calls():
    assert _convert_to_ollama_tool_calls([
        {
            'call_id': 'abc',
            'call_type': 'function',
            'function': {
                'name': 'foo',
                'arguments': '{"a": 7}',
            },
        },
        {
            'call_id': 'xyz',
            'call_type': 'function',
            'function': {
                'name': 'bar',
                'arguments': '{"b": 42}',
            },
        },
    ]) == [
        {
            'function': {
                'name': 'foo',
                'arguments': {
                    'a': 7,
                },
            },
        },
        {
            'function': {
                'name': 'bar',
                'arguments': {
                    'b': 42,
                },
            },
        },
    ]


@pytest.mark.asyncio
async def test_convert_to_ollama_messages():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'a.png')
        with open(fn, 'wb') as f:
            f.write(b'1234')
        assert await _convert_to_ollama_messages([
            {
                'role': 'system',
                'text': 'You are a robot.',
            },
            {
                'role': 'human',
                'text': 'What is in this image?',
                'media': [
                    {
                        'type': 'image',
                        'image': fn,
                    },
                ],
            },
            {
                'role': 'ai',
                'text': 'Nothing!',
            },
            {
                'role': 'tool_call',
                'tools': [
                    {
                        'call_id': 'abc',
                        'call_type': 'function',
                        'function': {
                            'name': 'foo',
                            'arguments': '{"a": 7}',
                        },
                    },
                    {
                        'call_id': 'xyz',
                        'call_type': 'function',
                        'function': {
                            'name': 'bar',
                            'arguments': '{"b": 42}',
                        },
                    },
                ],
            },
            {
                'role': 'tool_res',
                'tools': [
                    {
                        'call_id': 'xyz',
                        'type': 'any',
                        'result': True,
                    },
                    {
                        'call_id': 'abc',
                        'type': 'any',
                        'result': 'it worked',
                    },
                    {
                        'call_id': 'dne',
                        'type': 'any',
                        'result': 'ghost result',
                    },
                ],
            },
        ]) == [
            {
                'role': 'system',
                'content': 'You are a robot.',
            },
            {
                'role': 'user',
                'content': 'What is in this image?',
                'images': ['MTIzNA=='],
            },
            {
                'role': 'assistant',
                'content': 'Nothing!',
            },
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {
                        'function': {
                            'name': 'foo',
                            'arguments': {
                                'a': 7,
                            },
                        },
                    },
                    {
                        'function': {
                            'name': 'bar',
                            'arguments': {
                                'b': 42,
                            },
                        },
                    },
                ],
            },
            {
                'role': 'tool',
                'content': 'True',
                'name': 'bar',
            },
            {
                'role': 'tool',
                'content': 'it worked',
                'name': 'foo',
            },
            {
                'role': 'tool',
                'content': 'ghost result',
                'name': 'unknown',
            },
        ]


def test_set_cost_raw():
    orig_message: Message = {
        'role': 'ai',
        'text': 'Hi',
    }
    assert _set_cost_raw(orig_message, _sample_events_streaming_text) == {
        'role': 'ai',
        'text': 'Hi',
        'cost': {
            'input_tokens': 20,
            'output_tokens': 10,
            'total_tokens': 30,
        },
        'raw': _sample_events_streaming_text,
    }
    assert orig_message == {  # <-- orig message should be unmodified!
        'role': 'ai',
        'text': 'Hi',
    }


@pytest.mark.asyncio
async def test_process_stream_streaming_text():
    events = []
    async def callback(event: EventPayload) -> None:
        events.append(event)
    messages = await _process_stream(
        fake_async(_sample_events_streaming_text),
        callback,
    )
    assert events == [
        ('ai', 'text_event', 'Hello'),
        ('ai', 'text_event', '!'),
        ('ai', 'text_event', ' How'),
        ('ai', 'text_event', ' can'),
        ('ai', 'text_event', ' I'),
        ('ai', 'text_event', ' assist'),
        ('ai', 'text_event', ' you'),
        ('ai', 'text_event', ' today'),
        ('ai', 'text_event', '?'),
    ]
    assert messages == [
        {
            'role': 'ai',
            'text': 'Hello! How can I assist you today?',
            'cost': {
                'input_tokens': 20,
                'output_tokens': 10,
                'total_tokens': 30,
            },
            'raw': _sample_events_streaming_text,
        },
    ]


@pytest.mark.asyncio
async def test_process_stream_tool_call():
    events = []
    async def callback(event: EventPayload) -> None:
        events.append(event)
    messages = await _process_stream(
        fake_async(_sample_events_tool_call),
        callback,
    )
    tool_call: ToolCall = {
        'call_id': 'call_0',
        'call_type': 'function',
        'function': {
            'name': 'evaluate_math_expression',
            'arguments': '{"expression": "2.5 * sin(4.5)"}',
        },
    }
    assert events == [
        ('tool_call', 'text_event', 'evaluate_math_expression({"expression": "2.5 * sin(4.5)"})\n'),
        ('tool_call', 'tool_call_event', tool_call),
    ]
    assert messages == [
        {
            'role': 'tool_call',
            'tools': [tool_call],
            'cost': {
                'input_tokens': 100,
                'output_tokens': 29,
                'total_tokens': 129,
            },
            'raw': _sample_events_tool_call,
        },
    ]


@pytest.mark.asyncio
async def test_process_stream_text_and_tool_call():
    _sample_events = [
        *_sample_events_streaming_text,
        *_sample_events_tool_call,
    ]
    events = []
    async def callback(event: EventPayload) -> None:
        events.append(event)
    messages = await _process_stream(
        fake_async(_sample_events),
        callback,
    )
    tool_call: ToolCall = {
        'call_id': 'call_0',
        'call_type': 'function',
        'function': {
            'name': 'evaluate_math_expression',
            'arguments': '{"expression": "2.5 * sin(4.5)"}',
        },
    }
    assert events == [
        ('ai', 'text_event', 'Hello'),
        ('ai', 'text_event', '!'),
        ('ai', 'text_event', ' How'),
        ('ai', 'text_event', ' can'),
        ('ai', 'text_event', ' I'),
        ('ai', 'text_event', ' assist'),
        ('ai', 'text_event', ' you'),
        ('ai', 'text_event', ' today'),
        ('ai', 'text_event', '?'),
        ('tool_call', 'text_event', 'evaluate_math_expression({"expression": "2.5 * sin(4.5)"})\n'),
        ('tool_call', 'tool_call_event', tool_call),
    ]
    assert messages == [
        {
            'role': 'ai',
            'text': 'Hello! How can I assist you today?',
        },
        {
            'role': 'tool_call',
            'tools': [tool_call],
            'cost': {
                'input_tokens': 20 + 100,
                'output_tokens': 10 + 29,
                'total_tokens': 30 + 129,
            },
            'raw': _sample_events,
        },
    ]


@pytest.mark.asyncio
async def test_process_stream_structured_output():
    events = []
    async def callback(event: EventPayload) -> None:
        events.append(event)
    messages = await _process_stream(
        fake_async(_sample_events_structued_output),
        _wrap_event_callback_convert_ai_text_to_tool_call_text(callback),
    )
    assert events == [
        ('tool_call', 'text_event', '{"base":2,"exponent":101}'),
    ]
    assert messages == [
        {
            'role': 'ai',
            'text': '{"base":2,"exponent":101}',
            'cost': {
                'input_tokens': 28,
                'output_tokens': 15,
                'total_tokens': 43,
            },
            'raw': _sample_events_structued_output,
        },
    ]


class MyTypedDict(TypedDict):
    """My special type which is a TypedDict"""
    a: str
    b: int
    c: Literal['one', 'two', 'three']

class MyPydanticType(BaseModel):
    """My special type which is a Pydantic thing"""
    a: str
    b: int
    c: Literal['one', 'two', 'three']
    d: MyTypedDict

def test_get_ollama_format_for_structured_output():
    assert _get_ollama_format_for_structured_output(MyTypedDict) == {
        'type': 'object',
        'title': 'MyTypedDict',
        'description': 'My special type which is a TypedDict',
        'properties': {
            'a': {
                'type': 'string',
                'title': 'A',
            },
            'b': {
                'type': 'integer',
                'title': 'B',
            },
            'c': {
                'type': 'string',
                'title': 'C',
                'enum': ['one', 'two', 'three'],
            },
        },
        'required': ['a', 'b', 'c'],
        'additionalProperties': False,
    }
    assert _get_ollama_format_for_structured_output(MyPydanticType) == {
        'type': 'object',
        'title': 'MyPydanticType',
        'description': 'My special type which is a Pydantic thing',
        'properties': {
            'a': {
                'type': 'string',
                'title': 'A',
            },
            'b': {
                'type': 'integer',
                'title': 'B',
            },
            'c': {
                'type': 'string',
                'title': 'C',
                'enum': ['one', 'two', 'three'],
            },
            'd': {
                '$ref': '#/$defs/MyTypedDict',
            },
        },
        'required': ['a', 'b', 'c', 'd'],
        'additionalProperties': False,
        '$defs': {
            'MyTypedDict': {
                'type': 'object',
                'title': 'MyTypedDict',
                'description': 'My special type which is a TypedDict',
                'properties': {
                    'a': {
                        'type': 'string',
                        'title': 'A',
                    },
                    'b': {
                        'type': 'integer',
                        'title': 'B',
                    },
                    'c': {
                        'type': 'string',
                        'title': 'C',
                        'enum': ['one', 'two', 'three'],
                    },
                },
                'required': ['a', 'b', 'c'],
                'additionalProperties': False,
            },
        },
    }


@pytest.mark.asyncio
async def test_LasagnaOllama_run_once_invalid_arguments():
    async def callback(event: EventPayload) -> None:
        assert event

    with pytest.raises(ValueError, match=r"Oops! You cannot do both tool-use and structured output at the same time!"):
        await LasagnaOllama(model='mistral-small')._run_once(
            event_callback = callback,
            messages = [],
            tools_spec = _convert_to_ollama_tools([sample_tool]),
            force_tool = True,
            format = {'type': 'object'},
        )

    with pytest.raises(ValueError, match=r"Oops! Ollama currently does not support \*optional\* tool use."):
        await LasagnaOllama(model='mistral-small')._run_once(
            event_callback = callback,
            messages = [],
            tools_spec = _convert_to_ollama_tools([sample_tool]),
            force_tool = False,
            format = None,
        )

    with pytest.raises(ValueError, match=r"Oops! You cannot force tools that are not specified!"):
        await LasagnaOllama(model='mistral-small')._run_once(
            event_callback = callback,
            messages = [],
            tools_spec = None,
            force_tool = True,
            format = None,
        )
