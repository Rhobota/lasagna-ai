import pytest

from lasagna.lasagna_anthropic import (
    _convert_to_anthropic_messages,
    _build_messages_from_anthropic_payload,
    _convert_to_anthropic_tools,
    _make_raw_event,
    _process_stream,
)

from lasagna.types import (
    Message,
)

from typing import List

from anthropic.types.message import Message as AnthropicMessage

import os
import tempfile


@pytest.mark.asyncio
async def test_convert_to_anthropic_messages():
    # With system prompt:
    messages: List[Message] = [
        {
            'role': 'system',
            'text': 'You are a robot.',
        },
        {
            'role': 'human',
            'text': 'Hi, what are you?',
        },
    ]
    system_prompt, anthropic_messages = await _convert_to_anthropic_messages(messages)
    assert system_prompt == 'You are a robot.'
    assert anthropic_messages == [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Hi, what are you?',
                },
            ],
        },
    ]

    # Without system prompt:
    messages: List[Message] = [
        {
            'role': 'ai',
            'text': 'Hi, what can I help you with today?',
        },
    ]
    system_prompt, anthropic_messages = await _convert_to_anthropic_messages(messages)
    assert system_prompt is None
    assert anthropic_messages == [
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': 'Hi, what can I help you with today?',
                },
            ],
        },
    ]

    # Media:
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'a.png')
        with open(fn, 'wb') as f:
            f.write(b'1234')
        messages = [
            {
                'role': 'human',
                'text': "Hi, here's a picture.",
                'media': [
                    {
                        'media_type': 'image',
                        'image': fn,
                    },
                ],
            },
        ]
        system_prompt, anthropic_messages = await _convert_to_anthropic_messages(messages)
        assert system_prompt is None
        assert anthropic_messages == [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': "Hi, here's a picture.",
                    },
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': 'image/png',
                            'data': 'MTIzNA==',
                        },
                    },
                ],
            },
        ]

    # Tool messages:
    messages: List[Message] = [
        {
            'role': 'tool_call',
            'tools': [
                {
                    'call_type': 'function',
                    'call_id': 'abcd',
                    'function': {
                        'name': 'get_stuff',
                        'arguments': '{"q": "stuff about cats"}',
                    },
                },
                {
                    'call_type': 'function',
                    'call_id': 'wxyz',
                    'function': {
                        'name': 'feed_cat',
                        'arguments': '{}',
                    },
                },
            ],
        },
        {
            'role': 'tool_res',
            'tools': [
                {
                    'call_id': 'abcd',
                    'result': 'here is your stuff',
                },
                {
                    'call_id': 'wxyz',
                    'result': 'Error: cat 404',
                    'is_error': True,
                },
            ],
        },
    ]
    system_prompt, anthropic_messages = await _convert_to_anthropic_messages(messages)
    assert system_prompt is None
    assert anthropic_messages == [
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'tool_use',
                    'id': 'abcd',
                    'name': 'get_stuff',
                    'input': {'q': 'stuff about cats'},
                },
                {
                    'type': 'tool_use',
                    'id': 'wxyz',
                    'name': 'feed_cat',
                    'input': {},
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': 'abcd',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'here is your stuff',
                        },
                    ],
                },
                {
                    'type': 'tool_result',
                    'tool_use_id': 'wxyz',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'Error: cat 404',
                        },
                    ],
                    'is_error': True,
                },
            ],
        },
    ]

    # Invalid role:
    messages: List[Message] = [
        {
            'role': 'invalid',  # type: ignore
            'text': 'Hi, what can I help you with today?',
        },
    ]
    with pytest.raises(ValueError):
        await _convert_to_anthropic_messages(messages)

    # Collapse logic (most basic):
    messages: List[Message] = [
        {
            'role': 'ai',
            'text': 'Hi, what can I help you with today?',
        },
        {
            'role': 'ai',
            'text': 'Here is more text.',
        },
    ]
    system_prompt, anthropic_messages = await _convert_to_anthropic_messages(messages)
    assert system_prompt is None
    assert anthropic_messages == [
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': 'Hi, what can I help you with today?',
                },
                {
                    'type': 'text',
                    'text': 'Here is more text.',
                },
            ],
        },
    ]

    # Collapse logic (ensure not to collapse in this case):
    messages: List[Message] = [
        {
            'role': 'ai',
            'text': 'Hi, what can I help you with today?',
        },
        {
            'role': 'human',
            'text': 'I need help with everything.',
        },
        {
            'role': 'ai',
            'text': 'Please be more specific.',
        },
    ]
    system_prompt, anthropic_messages = await _convert_to_anthropic_messages(messages)
    assert system_prompt is None
    assert anthropic_messages == [
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': 'Hi, what can I help you with today?',
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'I need help with everything.',
                },
            ],
        },
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': 'Please be more specific.',
                },
            ],
        },
    ]

    # Collapse logic (more complex):
    messages: List[Message] = [
        {
            'role': 'ai',
            'text': 'Hi, what can I help you with today?',
        },
        {
            'role': 'human',
            'text': 'I need help with everything.',
        },
        {
            'role': 'human',
            'text': 'EVERYTHING!',
        },
        {
            'role': 'ai',
            'text': 'Please be more specific.',
        },
        {
            'role': 'ai',
            'text': 'Like tell me what you are trying to do.',
        },
    ]
    system_prompt, anthropic_messages = await _convert_to_anthropic_messages(messages)
    assert system_prompt is None
    assert anthropic_messages == [
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': 'Hi, what can I help you with today?',
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'I need help with everything.',
                },
                {
                    'type': 'text',
                    'text': 'EVERYTHING!',
                },
            ],
        },
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': 'Please be more specific.',
                },
                {
                    'type': 'text',
                    'text': 'Like tell me what you are trying to do.',
                },
            ],
        },
    ]

    # Collapse logic (more complex again):
    messages: List[Message] = [
        {
            'role': 'ai',
            'text': 'Hi, what can I help you with today?',
        },
        {
            'role': 'human',
            'text': 'I need help with everything.',
        },
        {
            'role': 'human',
            'text': 'EVERYTHING!',
        },
        {
            'role': 'ai',
            'text': 'Please be more specific.',
        },
        {
            'role': 'ai',
            'text': 'Like tell me what you are trying to do.',
        },
        {
            'role': 'human',
            'text': 'Forget it...',
        },
    ]
    system_prompt, anthropic_messages = await _convert_to_anthropic_messages(messages)
    assert system_prompt is None
    assert anthropic_messages == [
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': 'Hi, what can I help you with today?',
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'I need help with everything.',
                },
                {
                    'type': 'text',
                    'text': 'EVERYTHING!',
                },
            ],
        },
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': 'Please be more specific.',
                },
                {
                    'type': 'text',
                    'text': 'Like tell me what you are trying to do.',
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Forget it...',
                },
            ],
        },
    ]


@pytest.mark.asyncio
async def test_build_messages_from_anthropic_payload():
    raw = {
        "id": "msg_01NvDCPEEZgGEzpvYnjqru9n",
        "content": [
            {
                "text": "Hello, I'm Bob. Let me help you with those calculations:",
                "type": "text",
            },
            {
                "id": "toolu_011nFpn7UokLDMfiP6UW7DuJ",
                "input": {
                    "a": 5,
                    "b": 81,
                },
                "name": "multiply",
                "type": "tool_use",
            },
            {
                "id": "toolu_01VhUoeyXmswsFtEc8piHi4b",
                "input": {
                    "a": 89,
                    "b": 412,
                },
                "name": "multiply",
                "type": "tool_use",
            },
        ],
        "model": "claude-3-haiku-20240307",
        "role": "assistant",
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "type": "message",
        "usage": {
            "input_tokens": 508,
            "output_tokens": 82,
        },
    }
    anthropic_message = AnthropicMessage.model_validate(raw)
    messages = _build_messages_from_anthropic_payload([], anthropic_message)
    assert messages == [
        {
            'role': 'ai',
            'text': "Hello, I'm Bob. Let me help you with those calculations:",
        },
        {
            'role': 'tool_call',
            'tools': [
                {
                    'call_type': 'function',
                    'call_id': 'toolu_011nFpn7UokLDMfiP6UW7DuJ',
                    'function': {
                        'name': 'multiply',
                        'arguments': '{"a": 5, "b": 81}',
                    },
                },
                {
                    'call_type': 'function',
                    'call_id': 'toolu_01VhUoeyXmswsFtEc8piHi4b',
                    'function': {
                        'name': 'multiply',
                        'arguments': '{"a": 89, "b": 412}',
                    },
                },
            ],
            'cost': {
                'input_tokens': 508,
                'output_tokens': 82,
                'total_tokens': 590,
            },
            'raw': {
                'events': [],
                'message': raw,
            },
        },
    ]
