import pytest

from lasagna.types import (
    ChatMessage,
    ChatMessageRole,
    ToolCall,
    ToolResult,
    EventPayload,
)

from lasagna.stream import fake_async

from lasagna.lasagna_openai import (
    _extract_deltas,
    _process_output_stream,
    _convert_to_openai_tools,
    _convert_to_openai_messages,
    _build_messages_from_openai_payload,
    _handle_tools,
    _build_tool_response_message,
)

from typing import List, Dict, Callable

from openai.types.chat import (
    ChatCompletionChunk,
)


SAMPLE_TEXT_STREAM: List[ChatCompletionChunk] = [
    ChatCompletionChunk.model_validate(v)
    for v in [
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": "",
                        "role": "assistant"
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": "Hello"
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": " Ryan"
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": "!"
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": " "
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": "5"
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": " multiplied"
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": " by"
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": " "
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": "7"
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": " is"
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": " "
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": "35"
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {
                        "content": "."
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaHQcqZ177ddacDIf2nrdzbjoJRH",
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713739736,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        }
    ]
]


SAMPLE_TOOL_STREAM: List[ChatCompletionChunk] = [
    ChatCompletionChunk.model_validate(v)
    for v in [
        {
            "id": "chatcmpl-9GaOnH7T5O8VitWZryKt1qWy77GW8",
            "choices": [
                {
                    "delta": {
                        "content": None,
                        "role": "assistant"
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713740193,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaOnH7T5O8VitWZryKt1qWy77GW8",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_x7zmzwKI0LrwDF2xVMcfzXzN",
                                "function": {
                                    "arguments": "",
                                    "name": "multiply"
                                },
                                "type": "function"
                            }
                        ]
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713740193,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaOnH7T5O8VitWZryKt1qWy77GW8",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": "{\"a\""
                                }
                            }
                        ]
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713740193,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaOnH7T5O8VitWZryKt1qWy77GW8",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": ": 5, "
                                }
                            }
                        ]
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713740193,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaOnH7T5O8VitWZryKt1qWy77GW8",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": "\"b\": 7"
                                }
                            }
                        ]
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713740193,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaOnH7T5O8VitWZryKt1qWy77GW8",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": "}"
                                }
                            }
                        ]
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713740193,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaOnH7T5O8VitWZryKt1qWy77GW8",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_33vMBGeVd96A9BhW6H3r8jHb",
                                "function": {
                                    "arguments": "",
                                    "name": "multiply"
                                },
                                "type": "function"
                            }
                        ]
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713740193,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaOnH7T5O8VitWZryKt1qWy77GW8",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "function": {
                                    "arguments": "{\"a\""
                                }
                            }
                        ]
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713740193,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaOnH7T5O8VitWZryKt1qWy77GW8",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "function": {
                                    "arguments": ": 8, "
                                }
                            }
                        ]
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713740193,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaOnH7T5O8VitWZryKt1qWy77GW8",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "function": {
                                    "arguments": "\"b\": 1"
                                }
                            }
                        ]
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713740193,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaOnH7T5O8VitWZryKt1qWy77GW8",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "function": {
                                    "arguments": "01}"
                                }
                            }
                        ]
                    },
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713740193,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        },
        {
            "id": "chatcmpl-9GaOnH7T5O8VitWZryKt1qWy77GW8",
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "logprobs": None
                }
            ],
            "created": 1713740193,
            "model": "gpt-3.5-turbo-0125",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_c2295e73ad"
        }
    ]
]


CORRECT_PARSED_TOOLS: List[ToolCall] = [
    {
        "call_id": "call_x7zmzwKI0LrwDF2xVMcfzXzN",
        "function": {
            "arguments": "{\"a\": 5, \"b\": 7}",
            "name": "multiply"
        },
        "call_type": "function"
    },
    {
        "call_id": "call_33vMBGeVd96A9BhW6H3r8jHb",
        "function": {
            "arguments": "{\"a\": 8, \"b\": 101}",
            "name": "multiply"
        },
        "call_type": "function"
    }
]


@pytest.mark.asyncio
async def test_extract_deltas():
    stream = fake_async(SAMPLE_TEXT_STREAM[:2] + SAMPLE_TEXT_STREAM[-1:])
    vals = [(d.to_dict(), s) async for d, s in _extract_deltas(stream)]
    assert vals == [
        ({'content': '', 'role': 'assistant'}, None),
        ({'content': 'Hello'}, None),
        ({}, 'stop'),
    ]

    stream = fake_async(SAMPLE_TOOL_STREAM[:2] + SAMPLE_TOOL_STREAM[-1:])
    vals = [(d.to_dict(), s) async for d, s in _extract_deltas(stream)]
    assert vals == [
        ({'content': None, 'role': 'assistant'}, None),
        ({'tool_calls': [{
            'index': 0,
            'id': 'call_x7zmzwKI0LrwDF2xVMcfzXzN',
            'function': {'arguments': '', 'name': 'multiply'},
            'type': 'function',
        }]}, None),
        ({}, 'tool_calls'),
    ]


@pytest.mark.asyncio
async def test_process_output_stream__text():
    stream = _process_output_stream(_extract_deltas(fake_async(SAMPLE_TEXT_STREAM)))
    texts: List[str] = []
    async for role, type_, text in stream:
        assert role == ChatMessageRole.AI
        assert type_ == 'text'
        assert isinstance(text, str)
        texts.append(text)
    text = ''.join(texts)
    assert text == 'Hello Ryan! 5 multiplied by 7 is 35.'


@pytest.mark.asyncio
async def test_process_output_stream__tool():
    stream = _process_output_stream(_extract_deltas(fake_async(SAMPLE_TOOL_STREAM)))
    texts: List[str] = []
    tool_calls: List[ToolCall] = []
    async for event in stream:
        assert event[0] == ChatMessageRole.TOOL_CALL
        if event[1] == 'text':
            texts.append(event[2])
        elif event[1] == 'tool_call':
            tool_calls.append(event[2])
        else:
            assert False, event[1]
    text = ''.join(texts)
    assert text == 'multiply({"a": 5, "b": 7})\nmultiply({"a": 8, "b": 101})'
    assert tool_calls == CORRECT_PARSED_TOOLS

    stream = _process_output_stream(_extract_deltas(fake_async(SAMPLE_TOOL_STREAM[:6])))
    texts: List[str] = []
    tool_calls: List[ToolCall] = []
    async for event in stream:
        assert event[0] == ChatMessageRole.TOOL_CALL
        if event[1] == 'text':
            texts.append(event[2])
        elif event[1] == 'tool_call':
            tool_calls.append(event[2])
        else:
            assert False, event[1]
    text = ''.join(texts)
    assert text == 'multiply({"a": 5, "b": 7})'
    assert tool_calls == CORRECT_PARSED_TOOLS[:1]


@pytest.mark.asyncio
async def test_process_output_stream__text_and_tool():
    # The model can start with text and switch to tools!
    stream = _process_output_stream(_extract_deltas(fake_async(
        SAMPLE_TEXT_STREAM[:-1] + SAMPLE_TOOL_STREAM[1:]
    )))
    texts: List[str] = []
    tool_calls: List[ToolCall] = []
    async for event in stream:
        if event[0] == ChatMessageRole.AI:
            assert event[1] == 'text'
            texts.append(event[2])
        elif event[0] == ChatMessageRole.TOOL_CALL:
            if event[1] == 'text':
                texts.append(event[2])
            elif event[1] == 'tool_call':
                tool_calls.append(event[2])
            else:
                assert False, event[1]
        else:
            assert False, event[0]
    text = ''.join(texts)
    assert text == 'Hello Ryan! 5 multiplied by 7 is 35.\n\nmultiply({"a": 5, "b": 7})\nmultiply({"a": 8, "b": 101})'
    assert tool_calls == CORRECT_PARSED_TOOLS


def test_convert_to_openai_tools():
    x = 5
    def mytool(a, b):
        """
        Does a thing.
        :param: a: str: what to do
        :param: b: int: when to do it
        """
        return a, b, x
    def my_other_tool(how):
        """Something else
        that does stuff.

        :param: how: enum good bad: whether to do the thing in a good or bad way
        """
        return how
    def thing_with_optional_params(first, second=True, third=1):
        """This tool has optional params.
        :param: first: float: a number of sorts
        :param: second: bool: (optional) pass if you want
        :param: third: int: (optional) something else that's optional
        """
        return first, second, third
    spec = _convert_to_openai_tools([mytool, my_other_tool, thing_with_optional_params])
    assert spec
    assert len(spec) == 3
    assert spec[0] == {
        'type': 'function',
        'function': {
            'name': 'mytool',
            'description': 'Does a thing.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'a': {
                        'type': 'string',
                        'description': 'what to do',
                    },
                    'b': {
                        'type': 'integer',
                        'description': 'when to do it',
                    },
                },
                'required': ['a', 'b'],
            },
        },
    }
    assert spec[1] == {
        'type': 'function',
        'function': {
            'name': 'my_other_tool',
            'description': 'Something else that does stuff.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'how': {
                        'type': 'string',
                        'enum': ['good', 'bad'],
                        'description': 'whether to do the thing in a good or bad way',
                    },
                },
                'required': ['how'],
            },
        },
    }
    assert spec[2] == {
        'type': 'function',
        'function': {
            'name': 'thing_with_optional_params',
            'description': 'This tool has optional params.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'first': {
                        'type': 'number',
                        'description': 'a number of sorts',
                    },
                    'second': {
                        'type': 'boolean',
                        'description': '(optional) pass if you want',
                    },
                    'third': {
                        'type': 'integer',
                        'description': '(optional) something else that\'s optional',
                    },
                },
                'required': ['first'],
            },
        },
    }


@pytest.mark.asyncio
async def test_convert_to_openai_messages():
    messages: List[ChatMessage] = [
        {
            'role': 'INVALID', # type: ignore
            'text': 'bla',
            'cost': None,
            'raw': None,
        },
    ]
    with pytest.raises(ValueError):
        await _convert_to_openai_messages(messages)

    messages: List[ChatMessage] = [
        {'role': ChatMessageRole.SYSTEM, 'text': 'be nice', 'media': None, 'cost': None, 'raw': None},
        {'role': ChatMessageRole.HUMAN, 'text': 'hi', 'media': None, 'cost': None, 'raw': None},
        {'role': ChatMessageRole.AI, 'text': 'oh hi', 'media': None, 'cost': None, 'raw': None},
        {'role': ChatMessageRole.HUMAN, 'text': 'here is a picture', 'media': [{'media_type': 'image', 'image': 'http://example.com/img.png'}], 'cost': None, 'raw': None},
        {'role': ChatMessageRole.AI, 'text': 'thanks!', 'media': None, 'cost': None, 'raw': None},
        {'role': ChatMessageRole.HUMAN, 'text': 'here are two', 'media': [{'media_type': 'image', 'image': 'http://example.com/img.png'}, {'media_type': 'image', 'image': 'http://example.com/img2.png'}], 'cost': None, 'raw': None},
        {'role': ChatMessageRole.AI, 'text': 'double thanks!', 'media': None, 'cost': None, 'raw': None},
    ]
    ms = await _convert_to_openai_messages(messages)
    assert ms == [
        {'role': 'system', 'content': 'be nice'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]},
        {'role': 'assistant', 'content': 'oh hi'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'here is a picture'}, {'type': 'image_url', 'image_url': {'url': 'http://example.com/img.png'}}]},
        {'role': 'assistant', 'content': 'thanks!'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'here are two'}, {'type': 'image_url', 'image_url': {'url': 'http://example.com/img.png'}}, {'type': 'image_url', 'image_url': {'url': 'http://example.com/img2.png'}}]},
        {'role': 'assistant', 'content': 'double thanks!'},
    ]

    messages: List[ChatMessage] = [{
        'role': ChatMessageRole.TOOL_CALL,
        'tools': [
            {'call_id': 'call_x7zmzwKI0LrwDF2xVMcfzXzN', 'function': {'arguments': '{"a": 5, "b": 7}', 'name': 'multiply'}, 'call_type': 'function'},
            {'call_id': 'call_33vMBGeVd96A9BhW6H3r8jHb', 'function': {'arguments': '{"a": 8, "b": 101}', 'name': 'multiply'}, 'call_type': 'function'},
        ],
        'cost': None,
        'raw': None,
    }]
    ms = await _convert_to_openai_messages(messages)
    assert ms == [{
        'role': 'assistant',
        'content': None,
        'tool_calls': [
            {'id': 'call_x7zmzwKI0LrwDF2xVMcfzXzN', 'function': {'arguments': '{"a": 5, "b": 7}', 'name': 'multiply'}, 'type': 'function'},
            {'id': 'call_33vMBGeVd96A9BhW6H3r8jHb', 'function': {'arguments': '{"a": 8, "b": 101}', 'name': 'multiply'}, 'type': 'function'},
        ],
    }]

    messages: List[ChatMessage] = [{
        'role': ChatMessageRole.TOOL_RES,
        'tools': [
            {'call_id': '1002', 'result': 10.8 },
            {'call_id': '1003', 'result': "hihi" },
        ],
        'cost': None,
        'raw': None,
    }]
    ms = await _convert_to_openai_messages(messages)
    assert ms == [
        {
            'role': 'tool',
            'content': "10.8",
            'tool_call_id': "1002",
        },
        {
            'role': 'tool',
            'content': "hihi",
            'tool_call_id': "1003",
        },
    ]

    messages: List[ChatMessage] = [
        {
            'role': ChatMessageRole.AI,
            'text': "I'll use my tools!",
            'media': None,
            'cost': None,
            'raw': None,
        },
        {
            'role': ChatMessageRole.TOOL_CALL,
            'tools': [
                {'call_id': 'call_x7zmzwKI0LrwDF2xVMcfzXzN', 'function': {'arguments': '{"a": 5, "b": 7}', 'name': 'multiply'}, 'call_type': 'function'},
                {'call_id': 'call_33vMBGeVd96A9BhW6H3r8jHb', 'function': {'arguments': '{"a": 8, "b": 101}', 'name': 'multiply'}, 'call_type': 'function'},
            ],
            'cost': None,
            'raw': None,
        },
    ]
    ms = await _convert_to_openai_messages(messages)
    assert ms == [{
        'role': 'assistant',
        'content': "I'll use my tools!",
        'tool_calls': [
            {'id': 'call_x7zmzwKI0LrwDF2xVMcfzXzN', 'function': {'arguments': '{"a": 5, "b": 7}', 'name': 'multiply'}, 'type': 'function'},
            {'id': 'call_33vMBGeVd96A9BhW6H3r8jHb', 'function': {'arguments': '{"a": 8, "b": 101}', 'name': 'multiply'}, 'type': 'function'},
        ],
    }]


def test_build_messages_from_openai_payload():
    events: List[EventPayload] = [
        (ChatMessageRole.AI, 'text', 'Hello'),
        (ChatMessageRole.AI, 'text', ' Ryan'),
    ]
    messages = _build_messages_from_openai_payload([], events)
    assert messages == [{
        'role': ChatMessageRole.AI,
        'text': 'Hello Ryan',
        'media': None,
        'cost': None,
        'raw': [],
    }]

    tool_calls: List[ToolCall] = [
        {'call_id': 'call_x7zmzwKI0LrwDF2xVMcfzXzN', 'function': {'arguments': '{"a": 5, "b": 7}', 'name': 'multiply'}, 'call_type': 'function'},
        {'call_id': 'call_33vMBGeVd96A9BhW6H3r8jHb', 'function': {'arguments': '{"a": 8, "b": 101}', 'name': 'multiply'}, 'call_type': 'function'},
    ]
    events: List[EventPayload] = [
        (ChatMessageRole.TOOL_CALL, 'text', 'multiply('),
        (ChatMessageRole.TOOL_CALL, 'text', '{"a"'),
        (ChatMessageRole.TOOL_CALL, 'text', ': 5, '),
        (ChatMessageRole.TOOL_CALL, 'text', '"b": 7'),
        (ChatMessageRole.TOOL_CALL, 'text', '}'),
        (ChatMessageRole.TOOL_CALL, 'text', ')\n'),
        (ChatMessageRole.TOOL_CALL, 'text', 'multiply('),
        (ChatMessageRole.TOOL_CALL, 'text', '{"a"'),
        (ChatMessageRole.TOOL_CALL, 'text', ': 8, '),
        (ChatMessageRole.TOOL_CALL, 'text', '"b": 1'),
        (ChatMessageRole.TOOL_CALL, 'text', '01}'),
        (ChatMessageRole.TOOL_CALL, 'text', ')'),
        (ChatMessageRole.TOOL_CALL, 'tool_call', tool_calls[0]),
        (ChatMessageRole.TOOL_CALL, 'tool_call', tool_calls[1]),
    ]
    messages = _build_messages_from_openai_payload([], events)
    assert messages == [{
        'role': ChatMessageRole.TOOL_CALL,
        'tools': [
            {'call_id': 'call_x7zmzwKI0LrwDF2xVMcfzXzN', 'function': {'arguments': '{"a": 5, "b": 7}', 'name': 'multiply'}, 'call_type': 'function'},
            {'call_id': 'call_33vMBGeVd96A9BhW6H3r8jHb', 'function': {'arguments': '{"a": 8, "b": 101}', 'name': 'multiply'}, 'call_type': 'function'},
        ],
        'cost': None,
        'raw': [],
    }]

    tool_calls: List[ToolCall] = [
        {'call_id': 'call_x7zmzwKI0LrwDF2xVMcfzXzN', 'function': {'arguments': '{"a": 5, "b": 7}', 'name': 'multiply'}, 'call_type': 'function'},
        {'call_id': 'call_33vMBGeVd96A9BhW6H3r8jHb', 'function': {'arguments': '{"a": 8, "b": 101}', 'name': 'multiply'}, 'call_type': 'function'},
    ]
    events: List[EventPayload] = [
        (ChatMessageRole.AI, 'text', 'Hello'),
        (ChatMessageRole.AI, 'text', ' Ryan'),
        (ChatMessageRole.TOOL_CALL, 'text', 'multiply('),
        (ChatMessageRole.TOOL_CALL, 'text', '{"a"'),
        (ChatMessageRole.TOOL_CALL, 'text', ': 5, '),
        (ChatMessageRole.TOOL_CALL, 'text', '"b": 7'),
        (ChatMessageRole.TOOL_CALL, 'text', '}'),
        (ChatMessageRole.TOOL_CALL, 'text', ')\n'),
        (ChatMessageRole.TOOL_CALL, 'text', 'multiply('),
        (ChatMessageRole.TOOL_CALL, 'text', '{"a"'),
        (ChatMessageRole.TOOL_CALL, 'text', ': 8, '),
        (ChatMessageRole.TOOL_CALL, 'text', '"b": 1'),
        (ChatMessageRole.TOOL_CALL, 'text', '01}'),
        (ChatMessageRole.TOOL_CALL, 'text', ')'),
        (ChatMessageRole.TOOL_CALL, 'tool_call', tool_calls[0]),
        (ChatMessageRole.TOOL_CALL, 'tool_call', tool_calls[1]),
    ]
    messages = _build_messages_from_openai_payload([], events)
    assert messages == [
        {
            'role': ChatMessageRole.AI,
            'text': 'Hello Ryan',
            'media': None,
            'cost': None,
            'raw': None,
        },
        {
            'role': ChatMessageRole.TOOL_CALL,
            'tools': [
                {'call_id': 'call_x7zmzwKI0LrwDF2xVMcfzXzN', 'function': {'arguments': '{"a": 5, "b": 7}', 'name': 'multiply'}, 'call_type': 'function'},
                {'call_id': 'call_33vMBGeVd96A9BhW6H3r8jHb', 'function': {'arguments': '{"a": 8, "b": 101}', 'name': 'multiply'}, 'call_type': 'function'},
            ],
            'cost': None,
            'raw': [],
        },
    ]


def tool_a(first, second, third=5):
    return first + second + third

def tool_b(x):
    return x * 2

def test_handle_tools():
    x = 4
    def tool_c():
        return x * 2

    tool_map: Dict[str, Callable] = {
        'tool_a': tool_a,
        'tool_b': tool_b,
        'tool_c': tool_c,
    }
    message: ChatMessage = {
        'role': ChatMessageRole.TOOL_CALL,
        'tools': [
            {'call_id': '1001', 'function': {'arguments': '{"x": 8}', 'name': 'tool_b'}, 'call_type': 'function'},
            {'call_id': '1002', 'function': {'arguments': '{"x": 5.4}', 'name': 'tool_b'}, 'call_type': 'function'},
            {'call_id': '1003', 'function': {'arguments': '{"x": "hi"}', 'name': 'tool_b'}, 'call_type': 'function'},
            {'call_id': '1004', 'function': {'arguments': '{}', 'name': 'tool_b'}, 'call_type': 'function'},
            {'call_id': '1005', 'function': {'arguments': '{"y": "hi"}', 'name': 'tool_b'}, 'call_type': 'function'},
            {'call_id': '1006', 'function': {'arguments': '{"x": 4, "y": "hi"}', 'name': 'tool_b'}, 'call_type': 'function'},
            {'call_id': '1007', 'function': {'arguments': '{"first": 5, "second": 7.5}', 'name': 'tool_a'}, 'call_type': 'function'},
            {'call_id': '1008', 'function': {'arguments': '{"first": 5, "second": 7.5, "third": 3}', 'name': 'tool_a'}, 'call_type': 'function'},
            {'call_id': '1009', 'function': {'arguments': '{"third": 99, "first": 5, "second": 7.5}', 'name': 'tool_a'}, 'call_type': 'function'},
            {'call_id': '1010', 'function': {'arguments': '{}', 'name': 'tool_a'}, 'call_type': 'function'},
            {'call_id': '1011', 'function': {'arguments': '{"first": 5}', 'name': 'tool_a'}, 'call_type': 'function'},
            {'call_id': '1012', 'function': {'arguments': '{}', 'name': 'tool_c'}, 'call_type': 'function'},
            {'call_id': '1013', 'function': {'arguments': '{}', 'name': 'tool_d'}, 'call_type': 'function'},
        ],
        'cost': None,
        'raw': None,
    }
    tool_results = _handle_tools([message], tool_map)
    assert tool_results is not None
    assert tool_results == [
        {'call_id': '1001', 'result': 16 },
        {'call_id': '1002', 'result': 10.8 },
        {'call_id': '1003', 'result': "hihi" },
        {'call_id': '1004', 'result': "TypeError: tool_b() missing 1 required positional argument: 'x'" },
        {'call_id': '1005', 'result': "TypeError: tool_b() got an unexpected keyword argument 'y'" },
        {'call_id': '1006', 'result': "TypeError: tool_b() got an unexpected keyword argument 'y'" },
        {'call_id': '1007', 'result': 17.5 },
        {'call_id': '1008', 'result': 15.5 },
        {'call_id': '1009', 'result': 111.5 },
        {'call_id': '1010', 'result': "TypeError: tool_a() missing 2 required positional arguments: 'first' and 'second'" },
        {'call_id': '1011', 'result': "TypeError: tool_a() missing 1 required positional argument: 'second'" },
        {'call_id': '1012', 'result': 8 },
        {'call_id': '1013', 'result': "KeyError: 'tool_d'" },
    ]


def test_build_tool_response_message():
    res: List[ToolResult] = [
        {'call_id': '1002', 'result': 10.8 },
        {'call_id': '1003', 'result': "hihi" },
    ]
    message = _build_tool_response_message(res)
    assert message == {
        'role': ChatMessageRole.TOOL_RES,
        'tools': res,
        'cost': None,
        'raw': None,
    }