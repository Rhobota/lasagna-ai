from .types import (
    Model,
    EventCallback,
    Message,
    ExtractionType,
)
from .util import recursive_hash

from typing import Callable, List, Type, Tuple

from pydantic import BaseModel


class OpitonalToolCall(BaseModel):
    pass  # TODO


class ForcedToolCall(BaseModel):
    pass  # TODO


class ToolCallAsStructuredOutput(Model):
    def __init__(self, wrapped_model: Model) -> None:
        self.wrapped_model = wrapped_model

    def config_hash(self) -> str:
        return recursive_hash(
            'ToolCallAsStructuredOutput',
            {
                'wrapped_model': self.wrapped_model.config_hash(),
            },
        )

    async def run(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools: List[Callable],
        force_tool: bool = False,
        max_tool_iters: int = 5,
    ) -> List[Message]:
        if not tools:
            # No tool calls... so this is a trivial case:
            return await self.wrapped_model.run(
                event_callback = event_callback,
                messages = messages,
                tools = tools,
                force_tool = force_tool,
                max_tool_iters = max_tool_iters,
            )

        if force_tool:
            return []   # TODO

        else:
            return []   # TODO

    async def extract(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        extraction_type: Type[ExtractionType],
    ) -> Tuple[Message, ExtractionType]:
        return await self.wrapped_model.extract(
            event_callback = event_callback,
            messages = messages,
            extraction_type = extraction_type,
        )
