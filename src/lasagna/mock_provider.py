from lasagna.types import (
    EventCallback,
    Message,
    Model,
    EventPayload,
)

from lasagna.util import recursive_hash

from typing import List, Dict, Any, Callable


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
            {
                'role': 'ai',
                'text': f"model: {self.model}",
                'cost': None,
                'raw': None,
            },
        ]
        for key in sorted(self.model_kwargs.keys()):
            val = self.model_kwargs[key]
            m: Message = {
                'role': 'human',
                'text': f"model_kwarg: {key} = {val}",
                'cost': None,
                'raw': None,
            }
            res.append(m)
        return res
