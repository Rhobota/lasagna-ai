from lasagna.types import (
    EventCallback,
    Message,
    Model,
    EventPayload,
    ExtractionType,
    ToolCall,
)

from lasagna.util import recursive_hash

from typing import List, Dict, Tuple, Any, Callable, Type

import json


class MockProvider(Model):
    def __init__(self, model: str, **model_kwargs: Dict[str, Any]):
        self.model = model
        self.model_kwargs = model_kwargs

    def config_hash(self) -> str:
        return recursive_hash(None, {
            'provider': 'mock',
            'model': self.model,
            'model_kwargs': self.model_kwargs,
        })

    async def run(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools: List[Callable],
        force_tool: bool = False,
        max_tool_iters: int = 5,
    ) -> List[Message]:
        event: EventPayload = 'ai', 'text_event', 'Hi!'
        await event_callback(event)
        res: List[Message] = [
            *messages,
            {
                'role': 'ai',
                'text': f"model: {self.model}",
            },
        ]
        for key in sorted(self.model_kwargs.keys()):
            val = self.model_kwargs[key]
            m: Message = {
                'role': 'human',
                'text': f"model_kwarg: {key} = {val}",
            }
            res.append(m)
        return res

    async def extract(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        extraction_type: Type[ExtractionType],
    ) -> Tuple[Message, ExtractionType]:
        toolcall: ToolCall = {
            'call_id': 'id123',
            'call_type': 'function',
            'function': {
                'name': 'f',
                'arguments': json.dumps(self.model_kwargs),
            },
        }
        event: EventPayload = 'tool_call', 'tool_call_event', toolcall
        await event_callback(event)

        parsed = extraction_type(**self.model_kwargs)

        message: Message = {
            'role': 'ai',
            'text': None,
        }

        return message, parsed
