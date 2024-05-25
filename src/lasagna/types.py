import abc

from typing import (
    TypedDict, Dict, List, Any, Optional,
    Callable, Tuple, Awaitable, Union, Protocol,
)


class AgentRecord(TypedDict):
    name: str
    factory: Callable[[LLM, EventCallback, List[ChatMessage]], Awaitable[List[ChatMessage]]]


class ModelRecord(TypedDict):
    formal_name: str
    display_name: str


class LlmFactory(Protocol):
    def __call__(self, model: str, **model_kwargs) -> LLM: ...


class ModelProviderRecord(TypedDict):
    name: str
    factory: LlmFactory
    models: List[ModelRecord]


