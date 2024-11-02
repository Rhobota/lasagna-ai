import pytest

from lasagna.types import (
    Message,
    ToolCall,
    EventPayload,
)

from lasagna.stream import fake_async

from lasagna.lasagna_openai import (
    _extract_deltas,
    _process_output_stream,
    _convert_to_openai_tools,
    _convert_to_openai_messages,
    _build_messages_from_openai_payload,
)

from typing import List

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
        },
        {
            'id': 'chatcmpl-9ViswdADZmieYlPfvYf7J2UgTgFC7',
            'choices': [],
            'created': 1717347734,
            'model': 'gpt-3.5-turbo-0125',
            'object': 'chat.completion.chunk',
            'system_fingerprint': None,
            'usage': {'completion_tokens': 22, 'prompt_tokens': 21, 'total_tokens': 43},
        },
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
    stream = fake_async(SAMPLE_TEXT_STREAM[:2] + SAMPLE_TEXT_STREAM[-2:])
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
        assert role == 'ai'
        assert type_ == 'text_event'
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
        assert event[0] == 'tool_call'
        if event[1] == 'text_event':
            texts.append(event[2])
        elif event[1] == 'tool_call_event':
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
        assert event[0] == 'tool_call'
        if event[1] == 'text_event':
            texts.append(event[2])
        elif event[1] == 'tool_call_event':
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
        SAMPLE_TEXT_STREAM[:-2] + SAMPLE_TOOL_STREAM[1:]
    )))
    texts: List[str] = []
    tool_calls: List[ToolCall] = []
    async for event in stream:
        if event[0] == 'ai':
            assert event[1] == 'text_event'
            texts.append(event[2])
        elif event[0] == 'tool_call':
            if event[1] == 'text_event':
                texts.append(event[2])
            elif event[1] == 'tool_call_event':
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
            'strict': True,
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
                'additionalProperties': False,
            },
        },
    }
    assert spec[1] == {
        'type': 'function',
        'function': {
            'name': 'my_other_tool',
            'description': 'Something else that does stuff.',
            'strict': True,
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
                'additionalProperties': False,
            },
        },
    }
    assert spec[2] == {
        'type': 'function',
        'function': {
            'name': 'thing_with_optional_params',
            'description': 'This tool has optional params.',
            'strict': True,
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
                'additionalProperties': False,
            },
        },
    }


@pytest.mark.asyncio
async def test_convert_to_openai_messages():
    messages: List[Message] = [
        {
            'role': 'INVALID', # type: ignore
            'text': 'bla',
            'cost': None,
            'raw': None,
        },
    ]
    with pytest.raises(ValueError):
        await _convert_to_openai_messages(messages)

    messages: List[Message] = [
        {'role': 'system', 'text': 'be nice', 'cost': None, 'raw': None},
        {'role': 'human', 'text': 'hi', 'cost': None, 'raw': None},
        {'role': 'ai', 'text': 'oh hi', 'cost': None, 'raw': None},
        {'role': 'human', 'text': 'here is a picture', 'media': [{'type': 'image', 'image': 'http://example.com/img.png'}], 'cost': None, 'raw': None},
        {'role': 'ai', 'text': 'thanks!', 'cost': None, 'raw': None},
        {'role': 'human', 'text': 'here are two', 'media': [{'type': 'image', 'image': 'http://example.com/img.png'}, {'type': 'image', 'image': 'http://example.com/img2.png'}], 'cost': None, 'raw': None},
        {'role': 'ai', 'text': 'double thanks!', 'cost': None, 'raw': None},
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

    messages: List[Message] = [{
        'role': 'tool_call',
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

    messages: List[Message] = [{
        'role': 'tool_res',
        'tools': [
            {'type': 'any', 'call_id': '1002', 'result': 10.8 },
            {'type': 'any', 'call_id': '1003', 'result': "hihi" },
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

    messages: List[Message] = [
        {
            'role': 'ai',
            'text': "I'll use my tools!",
            'cost': None,
            'raw': None,
        },
        {
            'role': 'tool_call',
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
        ('ai', 'text_event', 'Hello'),
        ('ai', 'text_event', ' Ryan'),
    ]
    messages = _build_messages_from_openai_payload([], events)
    assert messages == [{
        'role': 'ai',
        'text': 'Hello Ryan',
        'cost': None,
        'raw': [],
    }]

    tool_calls: List[ToolCall] = [
        {'call_id': 'call_x7zmzwKI0LrwDF2xVMcfzXzN', 'function': {'arguments': '{"a": 5, "b": 7}', 'name': 'multiply'}, 'call_type': 'function'},
        {'call_id': 'call_33vMBGeVd96A9BhW6H3r8jHb', 'function': {'arguments': '{"a": 8, "b": 101}', 'name': 'multiply'}, 'call_type': 'function'},
    ]
    events: List[EventPayload] = [
        ('tool_call', 'text_event', 'multiply('),
        ('tool_call', 'text_event', '{"a"'),
        ('tool_call', 'text_event', ': 5, '),
        ('tool_call', 'text_event', '"b": 7'),
        ('tool_call', 'text_event', '}'),
        ('tool_call', 'text_event', ')\n'),
        ('tool_call', 'text_event', 'multiply('),
        ('tool_call', 'text_event', '{"a"'),
        ('tool_call', 'text_event', ': 8, '),
        ('tool_call', 'text_event', '"b": 1'),
        ('tool_call', 'text_event', '01}'),
        ('tool_call', 'text_event', ')'),
        ('tool_call', 'tool_call_event', tool_calls[0]),
        ('tool_call', 'tool_call_event', tool_calls[1]),
    ]
    messages = _build_messages_from_openai_payload([], events)
    assert messages == [{
        'role': 'tool_call',
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
        ('ai', 'text_event', 'Hello'),
        ('ai', 'text_event', ' Ryan'),
        ('tool_call', 'text_event', 'multiply('),
        ('tool_call', 'text_event', '{"a"'),
        ('tool_call', 'text_event', ': 5, '),
        ('tool_call', 'text_event', '"b": 7'),
        ('tool_call', 'text_event', '}'),
        ('tool_call', 'text_event', ')\n'),
        ('tool_call', 'text_event', 'multiply('),
        ('tool_call', 'text_event', '{"a"'),
        ('tool_call', 'text_event', ': 8, '),
        ('tool_call', 'text_event', '"b": 1'),
        ('tool_call', 'text_event', '01}'),
        ('tool_call', 'text_event', ')'),
        ('tool_call', 'tool_call_event', tool_calls[0]),
        ('tool_call', 'tool_call_event', tool_calls[1]),
    ]
    messages = _build_messages_from_openai_payload(SAMPLE_TEXT_STREAM[-3:], events)
    assert messages == [
        {
            'role': 'ai',
            'text': 'Hello Ryan',
            'cost': None,
            'raw': None,
        },
        {
            'role': 'tool_call',
            'tools': [
                {'call_id': 'call_x7zmzwKI0LrwDF2xVMcfzXzN', 'function': {'arguments': '{"a": 5, "b": 7}', 'name': 'multiply'}, 'call_type': 'function'},
                {'call_id': 'call_33vMBGeVd96A9BhW6H3r8jHb', 'function': {'arguments': '{"a": 8, "b": 101}', 'name': 'multiply'}, 'call_type': 'function'},
            ],
            'cost': {
                'input_tokens': 21,
                'output_tokens': 22,
                'total_tokens': 43,
            },
            'raw': [
                v.to_dict()
                for v in SAMPLE_TEXT_STREAM[-3:]
            ],
        },
    ]
