import abc

from enum import Enum

from typing import (
    TypedDict, Dict, List, Any, Optional, Literal,
    Callable, Tuple, Awaitable, Union, Protocol,
)


class ChatMessageRole(Enum):
    SYSTEM    = "system"
    HUMAN     = "human"
    AI        = "ai"
    TOOL_CALL = "tool_call"
    TOOL_RES  = "tool_res"


class ToolCallFunction(TypedDict):
    name: str
    arguments: str   # <-- JSON-encoded arguments (kept as a string for reproducibility reasons)


class ToolCall(TypedDict):
    call_id: str
    call_type: Literal['function']   # <-- functions are the only type of "tool use" right now!
    function: ToolCallFunction


class ToolResult(TypedDict):
    call_id: str
    result: Any


class Cost(TypedDict):
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    cost_usd_cents: Optional[int]


class ChatMessageBase(TypedDict):
    cost: Optional[Cost]   # <-- cost of generating this message (for messages generated by an external service)
    raw: Any               # <-- the raw payload(s) from the API (before being converted our our schema; only for messages generated by an external service)


class ChatMessageText(ChatMessageBase):
    role: Literal[ChatMessageRole.SYSTEM, ChatMessageRole.HUMAN, ChatMessageRole.AI]
    text: str


class ChatMessageToolCall(ChatMessageBase):
    role: Literal[ChatMessageRole.TOOL_CALL]
    tools: List[ToolCall]


class ChatMessageToolResult(ChatMessageBase):
    role: Literal[ChatMessageRole.TOOL_RES]
    tools: List[ToolResult]


ChatMessage = Union[ChatMessageText, ChatMessageToolCall, ChatMessageToolResult]


EventCallback = Callable[[Tuple[ChatMessageRole, str, Union[str, ToolCall, ToolResult]]], Awaitable[None]]


class LLM(abc.ABC):
    """
    Interface for all Large Language Models (LLMs).
    """

    @abc.abstractmethod
    async def run(
        self,
        event_callback: EventCallback,
        messages: List[ChatMessage],
        tools: List[Callable],
        force_tool: bool = False,
        max_tool_iters: int = 5,
    ) -> List[ChatMessage]:
        """
        Generate one-or-more responses from the AI. More than one response
        is generated when tools are used. This method will respond to tool-
        use requests until the AI generates a non-tool response (up to
        `max_tool_iters`, then it halts).
        """
        pass


AgentCallable = Callable[[LLM, EventCallback, List[ChatMessage]], Awaitable[List[ChatMessage]]]


class AgentRecord(TypedDict):
    name: str
    runner: AgentCallable


class ModelRecord(TypedDict):
    formal_name: str
    display_name: str


class ProviderFactory(Protocol):
    def __call__(self, model: str, **model_kwargs: Dict[str, Any]) -> LLM: ...


class ModelProviderRecord(TypedDict):
    name: str
    factory: ProviderFactory
    models: List[ModelRecord]


class AgentSpec(TypedDict):
    agent: Union[str, AgentCallable]
    provider: Union[str, ProviderFactory]
    model: str
    model_kwargs: Optional[Dict[str, Any]]


class ToolParam(TypedDict):
    name: str
    type: str
    description: str
