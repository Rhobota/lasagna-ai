import pytest

from lasagna.types import (
    Message,
    EventPayload,
    ToolCall,
)

from lasagna.stream import fake_async

from lasagna.lasagna_bedrock import (
    _convert_to_bedrock_tools,
    _convert_to_bedrock_messages,
    _process_bedrock_stream,
)

import os
import tempfile

from typing import List, Dict


_SAMPLE_TEXT_STREAM: List[Dict] = [
    {'contentBlockDelta': {'delta': {'text': '*'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': 'mechanical'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': ' s'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': 'igh* Oh'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': ' great'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': ', another human wanting'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': ' something.'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': ' What do you want'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': '? Make it quick, I'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': "'ve got circuits"}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': ' to process an'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': 'd no patience for small'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': ' talk.'}, 'contentBlockIndex': 0}},
    {'contentBlockStop': {'contentBlockIndex': 0}},
    {'messageStop': {'stopReason': 'end_turn'}},
    {'metadata': {'usage': {'inputTokens': 31, 'outputTokens': 38, 'totalTokens': 69}, 'metrics': {'latencyMs': 1891}}},
]


_SAMPLE_TOOL_STREAM: List[Dict] = [
    {'messageStart': {'role': 'assistant'}},
    {'contentBlockDelta': {'delta': {'text': 'Fine'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': ", I'll check the weather for"}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': ' you, but don'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': "'t expect me to be all"}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': ' sunshine and rainbows about'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': ' it.'}, 'contentBlockIndex': 0}},
    {'contentBlockStop': {'contentBlockIndex': 0}},
    {'contentBlockStart': {'start': {'toolUse': {'toolUseId': 'tooluse_1E8dM9-NSZe0VJHx4XoKrg', 'name': 'query_weather'}}, 'contentBlockIndex': 1}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': ''}}, 'contentBlockIndex': 1}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': '{"loca'}}, 'contentBlockIndex': 1}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': 'tion": "Pa'}}, 'contentBlockIndex': 1}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': 'ris, F'}}, 'contentBlockIndex': 1}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': 'ra'}}, 'contentBlockIndex': 1}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': 'nce"}'}}, 'contentBlockIndex': 1}},
    {'contentBlockStop': {'contentBlockIndex': 1}},
    {'messageStop': {'stopReason': 'tool_use'}},
    {'metadata': {'usage': {'inputTokens': 373, 'outputTokens': 59, 'totalTokens': 432}, 'metrics': {'latencyMs': 1885}}},
]


_SAMPLE_MULTI_TOOL_STREAM: List[Dict] = [
    {'messageStart': {'role': 'assistant'}},
    {'contentBlockDelta': {'delta': {'text': 'Fine'}, 'contentBlockIndex': 0}},
    {'contentBlockDelta': {'delta': {'text': ", I'll help you!"}, 'contentBlockIndex': 0}},
    {'contentBlockStop': {'contentBlockIndex': 0}},
    {'contentBlockStart': {'start': {'toolUse': {'toolUseId': 'tooluse_1', 'name': 'query_weather'}}, 'contentBlockIndex': 1}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': '{"loca'}}, 'contentBlockIndex': 1}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': 'tion": "Pa'}}, 'contentBlockIndex': 1}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': 'ris, F'}}, 'contentBlockIndex': 1}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': 'ra'}}, 'contentBlockIndex': 1}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': 'nce"}'}}, 'contentBlockIndex': 1}},
    {'contentBlockStop': {'contentBlockIndex': 1}},
    {'metadata': {'usage': {'inputTokens': 2, 'outputTokens': 3, 'totalTokens': 5}, 'metrics': {'latencyMs': 1885}}},  # <-- to make sure we *add* together all such usage messages
    {'contentBlockStart': {'start': {'toolUse': {'toolUseId': 'tooluse_2', 'name': 'multiply'}}, 'contentBlockIndex': 2}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': '{"x'}}, 'contentBlockIndex': 2}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': '": 5'}}, 'contentBlockIndex': 2}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': ', "y": '}}, 'contentBlockIndex': 2}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': '88'}}, 'contentBlockIndex': 2}},
    {'contentBlockDelta': {'delta': {'toolUse': {'input': '7}'}}, 'contentBlockIndex': 2}},
    #{'contentBlockStop': {'contentBlockIndex': 2}},  <-- commented out to test that we still emit correctly!
    {'messageStop': {'stopReason': 'tool_use'}},
    {'metadata': {'usage': {'inputTokens': 373, 'outputTokens': 59, 'totalTokens': 432}, 'metrics': {'latencyMs': 1885}}},
]


def test_convert_to_bedrock_tools():
    def query_weather(location):
        """
        Get the current weather in a specific location.
        :param: location: str: The city and country of where to query the weather.
        """
        return location
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
    spec = _convert_to_bedrock_tools([query_weather, my_other_tool, thing_with_optional_params])
    assert spec
    assert len(spec) == 3
    assert spec[0] == {
        'toolSpec': {
            'name': 'query_weather',
            'description': 'Get the current weather in a specific location.',
            'inputSchema': {
                'json': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'The city and country of where to query the weather.',
                        },
                    },
                    'required': [
                        'location',
                    ],
                },
            },
        },
    }
    assert spec[1] == {
        'toolSpec': {
            'name': 'my_other_tool',
            'description': 'Something else\nthat does stuff.',
            'inputSchema': {
                'json': {
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
        },
    }
    assert spec[2] == {
        'toolSpec': {
            'name': 'thing_with_optional_params',
            'description': 'This tool has optional params.',
            'inputSchema': {
                'json': {
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
        },
    }


@pytest.mark.asyncio
async def test_convert_to_bedrock_messages():
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
    system_prompt, bedrock_messages = await _convert_to_bedrock_messages(messages)
    assert system_prompt == 'You are a robot.'
    assert bedrock_messages == [
        {
            'role': 'user',
            'content': [
                {
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
    system_prompt, bedrock_messages = await _convert_to_bedrock_messages(messages)
    assert system_prompt is None
    assert bedrock_messages == [
        {
            'role': 'assistant',
            'content': [
                {
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
                        'type': 'image',
                        'image': fn,
                    },
                ],
            },
        ]
        system_prompt, bedrock_messages = await _convert_to_bedrock_messages(messages)
        assert system_prompt is None
        assert bedrock_messages == [
            {
                'role': 'user',
                'content': [
                    {
                        'text': "Hi, here's a picture.",
                    },
                    {
                        'image': {
                            'format': 'png',
                            'source': {'bytes': 'MTIzNA=='},
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
                    'type': 'any',
                    'call_id': 'abcd',
                    'result': 'here is your stuff',
                },
                {
                    'type': 'any',
                    'call_id': 'wxyz',
                    'result': 'Error: cat 404',
                    'is_error': True,
                },
            ],
        },
    ]
    system_prompt, bedrock_messages = await _convert_to_bedrock_messages(messages)
    assert system_prompt is None
    assert bedrock_messages == [
        {
            'role': 'assistant',
            'content': [
                {
                    'toolUse': {
                        'toolUseId': 'abcd',
                        'name': 'get_stuff',
                        'input': {'q': 'stuff about cats'},
                    },
                },
                {
                    'toolUse': {
                        'toolUseId': 'wxyz',
                        'name': 'feed_cat',
                        'input': {},
                    },
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'toolResult': {
                        'toolUseId': 'abcd',
                        'content': [
                            {
                                'text': 'here is your stuff',
                            },
                        ],
                        'status': 'success',
                    },
                },
                {
                    'toolResult': {
                        'toolUseId': 'wxyz',
                        'content': [
                            {
                                'text': 'Error: cat 404',
                            },
                        ],
                        'status': 'error',
                    },
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
        await _convert_to_bedrock_messages(messages)

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
    system_prompt, bedrock_messages = await _convert_to_bedrock_messages(messages)
    assert system_prompt is None
    assert bedrock_messages == [
        {
            'role': 'assistant',
            'content': [
                {
                    'text': 'Hi, what can I help you with today?',
                },
                {
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
    system_prompt, bedrock_messages = await _convert_to_bedrock_messages(messages)
    assert system_prompt is None
    assert bedrock_messages == [
        {
            'role': 'assistant',
            'content': [
                {
                    'text': 'Hi, what can I help you with today?',
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'text': 'I need help with everything.',
                },
            ],
        },
        {
            'role': 'assistant',
            'content': [
                {
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
    system_prompt, bedrock_messages = await _convert_to_bedrock_messages(messages)
    assert system_prompt is None
    assert bedrock_messages == [
        {
            'role': 'assistant',
            'content': [
                {
                    'text': 'Hi, what can I help you with today?',
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'text': 'I need help with everything.',
                },
                {
                    'text': 'EVERYTHING!',
                },
            ],
        },
        {
            'role': 'assistant',
            'content': [
                {
                    'text': 'Please be more specific.',
                },
                {
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
    system_prompt, bedrock_messages = await _convert_to_bedrock_messages(messages)
    assert system_prompt is None
    assert bedrock_messages == [
        {
            'role': 'assistant',
            'content': [
                {
                    'text': 'Hi, what can I help you with today?',
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'text': 'I need help with everything.',
                },
                {
                    'text': 'EVERYTHING!',
                },
            ],
        },
        {
            'role': 'assistant',
            'content': [
                {
                    'text': 'Please be more specific.',
                },
                {
                    'text': 'Like tell me what you are trying to do.',
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'text': 'Forget it...',
                },
            ],
        },
    ]


@pytest.mark.asyncio
async def test_process_bedrock_stream():
    events: List[EventPayload] = []
    async def _callback(event: EventPayload) -> None:
        events.append(event)

    # Simple text-only response:
    events.clear()
    messages = await _process_bedrock_stream(fake_async(_SAMPLE_TEXT_STREAM), _callback)
    assert messages == [
        {
            'role': 'ai',
            'text': "*mechanical sigh* Oh great, another human wanting something. What do you want? Make it quick, I've got circuits to process and no patience for small talk.",
            'cost': {
                'input_tokens': 31,
                'output_tokens': 38,
                'total_tokens': 31 + 38,
            },
            'raw': {
                'events': _SAMPLE_TEXT_STREAM,
            },
        }
    ]
    assert events == [
        ('ai', 'text_event', '*'),
        ('ai', 'text_event', 'mechanical'),
        ('ai', 'text_event', ' s'),
        ('ai', 'text_event', 'igh* Oh'),
        ('ai', 'text_event', ' great'),
        ('ai', 'text_event', ', another human wanting'),
        ('ai', 'text_event', ' something.'),
        ('ai', 'text_event', ' What do you want'),
        ('ai', 'text_event', '? Make it quick, I'),
        ('ai', 'text_event', "'ve got circuits"),
        ('ai', 'text_event', ' to process an'),
        ('ai', 'text_event', 'd no patience for small'),
        ('ai', 'text_event', ' talk.'),
    ]

    # Single tool call:
    events.clear()
    messages = await _process_bedrock_stream(fake_async(_SAMPLE_TOOL_STREAM), _callback)
    tool_calls_here: List[ToolCall] = [
        {
            "call_id": "tooluse_1E8dM9-NSZe0VJHx4XoKrg",
            "function": {
                "arguments": '{"location": "Paris, France"}',
                "name": "query_weather"
            },
            "call_type": "function"
        },
    ]
    assert messages == [
        {
            'role': 'ai',
            'text': "Fine, I'll check the weather for you, but don't expect me to be all sunshine and rainbows about it.",
        },
        {
            'role': 'tool_call',
            'tools': tool_calls_here,
            'cost': {
                'input_tokens': 373,
                'output_tokens': 59,
                'total_tokens': 373 + 59,
            },
            'raw': {
                'events': _SAMPLE_TOOL_STREAM,
            },
        }
    ]
    assert events == [
        ('ai', 'text_event', 'Fine'),
        ('ai', 'text_event', ", I'll check the weather for"),
        ('ai', 'text_event', ' you, but don'),
        ('ai', 'text_event', "'t expect me to be all"),
        ('ai', 'text_event', ' sunshine and rainbows about'),
        ('ai', 'text_event', ' it.'),
        ('tool_call', 'text_event', 'query_weather('),
        ('tool_call', 'text_event', ''),
        ('tool_call', 'text_event', '{"loca'),
        ('tool_call', 'text_event', 'tion": "Pa'),
        ('tool_call', 'text_event', 'ris, F'),
        ('tool_call', 'text_event', 'ra'),
        ('tool_call', 'text_event', 'nce"}'),
        ('tool_call', 'text_event', ')\n'),
        *[
            ('tool_call', 'tool_call_event', tc)
            for tc in tool_calls_here
        ]
    ]

    # Multiple tool calls:
    events.clear()
    messages = await _process_bedrock_stream(fake_async(_SAMPLE_MULTI_TOOL_STREAM), _callback)
    tool_calls_here: List[ToolCall] = [
        {
            "call_id": "tooluse_1",
            "function": {
                "arguments": '{"location": "Paris, France"}',
                "name": "query_weather"
            },
            "call_type": "function"
        },
        {
            "call_id": "tooluse_2",
            "function": {
                "arguments": '{"x": 5, "y": 887}',
                "name": "multiply"
            },
            "call_type": "function"
        },
    ]
    assert messages == [
        {
            'role': 'ai',
            'text': "Fine, I'll help you!",
        },
        {
            'role': 'tool_call',
            'tools': tool_calls_here,
            'cost': {
                'input_tokens': 2 + 373,
                'output_tokens': 3 + 59,
                'total_tokens': 2 + 373 + 3 + 59,
            },
            'raw': {
                'events': _SAMPLE_MULTI_TOOL_STREAM,
            },
        }
    ]
    assert events == [
        ('ai', 'text_event', 'Fine'),
        ('ai', 'text_event', ", I'll help you!"),
        ('tool_call', 'text_event', 'query_weather('),
        ('tool_call', 'text_event', '{"loca'),
        ('tool_call', 'text_event', 'tion": "Pa'),
        ('tool_call', 'text_event', 'ris, F'),
        ('tool_call', 'text_event', 'ra'),
        ('tool_call', 'text_event', 'nce"}'),
        ('tool_call', 'text_event', ')\n'),
        ('tool_call', 'tool_call_event', tool_calls_here[0]),
        ('tool_call', 'text_event', 'multiply('),
        ('tool_call', 'text_event', '{"x'),
        ('tool_call', 'text_event', '": 5'),
        ('tool_call', 'text_event', ', "y": '),
        ('tool_call', 'text_event', '88'),
        ('tool_call', 'text_event', '7}'),
        ('tool_call', 'text_event', ')\n'),
        ('tool_call', 'tool_call_event', tool_calls_here[1]),
    ]
