from ..types import Model, EventCallback, Message, ExtractionType

from pydantic import BaseModel

import random
from typing import List, Tuple, Callable, Any, Type, cast


class DebugModel(Model):
    def __init__(self, model: str, **model_kwargs: Any) -> None:
        seed = model_kwargs.get('seed', 0)
        self.rand = random.Random()
        self.rand.seed(seed, version=2)

    async def run(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools: List[Callable],
        force_tool: bool = False,
        max_tool_iters: int = 5,
    ) -> List[Message]:
        assert len(tools) == 0
        assert not force_tool
        return [{
            'role': 'ai',
            'text': f'The answer is: {len(messages)}',
        }]

    async def extract(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        extraction_type: Type[ExtractionType],
    ) -> Tuple[Message, ExtractionType]:
        message: Message = {
            'role': 'ai',
            'text': f'Will extract from {len(messages)} messages...',
        }
        is_trivial = self.rand.choice([True, False])
        if len(messages) > 3:
            is_trivial = True  # don't go too deep!
        last_message = messages[-1]
        assert last_message['role'] == 'human'
        data = dict(
            thoughts = '',
            task_statement = last_message['text'] or '!!! NO TEXT !!!',
            is_trivial = is_trivial,
            subtasks = [] if is_trivial else self._rand_tasks(),
        )
        if issubclass(extraction_type, BaseModel):
            output = cast(ExtractionType, extraction_type.model_validate(data))
        else:
            raise ValueError(f'Cannot handle data for type: {extraction_type}')
        return message, output

    def _rand_tasks(self) -> List[str]:
        length = self.rand.randint(2, 5)  # inclusive!
        return [
            f'step #{i + 1}: bla bla bla'
            for i in range(length)
        ]

    def _text(self, messages: List[Message]) -> str:
        return ' || '.join(f"[[ {m['role']}: {m.get('text')} ]]" for m in messages)

    def config_hash(self) -> str:
        raise NotImplementedError('not needed and not implemented')
