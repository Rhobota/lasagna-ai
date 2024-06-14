from __future__ import annotations
import abc

from typing import (
    TypedDict, Dict, List, Any, Optional, Literal,
    Callable, Tuple, Awaitable, Union, Protocol,
)

from typing_extensions import NotRequired


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


class MediaImage(TypedDict):
    media_type: Literal['image']
    image: str   # <-- either a local filepath or a remote URL


Media = MediaImage   # note: this will become a Union[MediaImage, ...] in the future as needed


class MessageBase(TypedDict):
    cost: NotRequired[Optional[Cost]]  # <-- cost of generating this message (for messages generated by an external service)
    raw: NotRequired[Any]              # <-- the raw payload(s) from the API (before being converted our our schema; only for messages generated by an external service)


class MessageContent(MessageBase):
    role: Literal['system', 'human', 'ai']
    text: Optional[str]
    media: NotRequired[List[Media]]


class MessageToolCall(MessageBase):
    role: Literal['tool_call']
    tools: List[ToolCall]


class MessageToolResult(MessageBase):
    role: Literal['tool_res']
    tools: List[ToolResult]


Message = Union[MessageContent, MessageToolCall, MessageToolResult]


EventPayload = Union[
    Tuple[Literal['human'],     Literal['echo_event'],      MessageContent],
    Tuple[Literal['ai'],        Literal['text_event'],      str],
    Tuple[Literal['tool_call'], Literal['text_event'],      str],
    Tuple[Literal['tool_call'], Literal['tool_call_event'], ToolCall],
    Tuple[Literal['tool_res'],  Literal['tool_res_event'],  ToolResult],
    Tuple[Literal['progress'],  Literal['start'],           Tuple[str, str]],    # payload is `(key, details)`
    Tuple[Literal['progress'],  Literal['update'],          Tuple[str, float]],  # payload is `(key, progress_0_to_1)`
    Tuple[Literal['progress'],  Literal['end'],             str],                # payload is `key`
]


EventCallback = Callable[[EventPayload], Awaitable[None]]


class Model(abc.ABC):
    """
    Interface for an AI model.

    In many cases this will be an LLM (when you're in pure-text use-cases). But
    this interface also supports AI models that operate on multimodal content,
    thus we'll use the phrase "AI model" to be more general.
    """

    @abc.abstractmethod
    async def run(
        self,
        event_callback: EventCallback,
        messages: List[Message],
        tools: List[Callable],
        force_tool: bool = False,
        max_tool_iters: int = 5,
    ) -> List[Message]:
        """
        Generate one-or-more responses from the AI. More than one response
        is generated when tools are used. This method will respond to tool-
        use requests until the AI generates a non-tool response (up to
        `max_tool_iters`, then it halts).
        """
        pass


class AgentRunBase(TypedDict):
    agent: NotRequired[str]
    provider: NotRequired[str]
    model: NotRequired[str]
    model_kwargs: NotRequired[Dict[str, Any]]


class AgentRunMessageList(AgentRunBase):
    type: Literal['messages']
    messages: List[Message]


class AgentRunParallel(AgentRunBase):
    type: Literal['parallel']
    runs: List[AgentRun]


class AgentRunChained(AgentRunBase):
    type: Literal['chain']
    runs: List[AgentRun]


AgentRun = Union[AgentRunMessageList, AgentRunParallel, AgentRunChained]


AgentCallable = Callable[[Model, EventCallback, List[AgentRun]], Awaitable[AgentRun]]

BoundAgentCallable = Callable[[EventCallback, List[AgentRun]], Awaitable[AgentRun]]


class AgentRecord(TypedDict):
    name: str
    runner: AgentCallable


class ModelRecord(TypedDict):
    formal_name: str
    display_name: str


class ModelFactory(Protocol):
    def __call__(self, model: str, **model_kwargs: Dict[str, Any]) -> Model: ...


class ProviderRecord(TypedDict):
    name: str
    factory: ModelFactory
    models: List[ModelRecord]


class AgentSpec(TypedDict):
    agent: Union[str, AgentCallable]
    provider: Union[str, ModelFactory]
    model: str
    model_kwargs: NotRequired[Dict[str, Any]]


class ToolParam(TypedDict):
    name: str
    type: str
    description: str


class CacheEventPayload(TypedDict):
    delta_time: float   # <-- when this event arrived (as the number of second since the start of the agent's execution)
    event: EventPayload


CacheKey = str


class CacheRecord(TypedDict):
    events: List[CacheEventPayload]
    run: AgentRun
