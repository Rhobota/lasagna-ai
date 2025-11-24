from lasagna.types import (
    EventCallback,
    Message,
    Model,
    ExtractionType,
)

from lasagna.util import recursive_hash

from typing import List, Tuple, Any, Callable, Type


class AbstractModel(Model):
    def __init__(self, model: str, **model_kwargs: Any):
        assert not model
        assert not model_kwargs

    def config_hash(self) -> str:
        return recursive_hash(None, {
            'provider': '__abstract__',
        })

    async def run(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools: List[Callable],
        force_tool: bool = False,
        max_tool_iters: int = 5,
    ) -> List[Message]:
        raise NotImplementedError('do not call the abstract model!')

    async def extract(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        extraction_type: Type[ExtractionType],
    ) -> Tuple[Message, ExtractionType]:
        raise NotImplementedError('do not call the abstract model!')
