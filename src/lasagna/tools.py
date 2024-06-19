import asyncio
import inspect
import json

from typing import List, Dict, Callable, Union, Any

from .types import (
    Message,
    ToolCall,
    ToolResult,
    ToolParam,
)


def convert_to_json_schema(params: List[ToolParam]) -> Dict[str, object]:
    def convert_type(t: str) -> Dict[str, object]:
        if t.startswith('enum '):
            return {
                "type": "string",
                "enum": t.split()[1:],
            }
        else:
            return {
                "type": {
                    'str': 'string',
                    'float': 'number',
                    'int': 'integer',
                    'bool': 'boolean',
                }[t],
            }
    return {
        "type": "object",
        "properties": {
            p['name']: {
                **convert_type(p['type']),
                "description": p['description'],
            }
            for p in params
        },
        "required": [
            p['name']
            for p in params
            if not p['description'].startswith('(optional)')
        ],
    }


async def handle_tools(
    messages: List[Message],
    tools_map: Dict[str, Callable],
) -> Union[List[ToolResult], None]:
    assert len(messages) > 0
    message = messages[-1]   # <-- the tool message will be last, if at all TODO: is this true for Anthropic?
    if message['role'] != 'tool_call':
        return None
    to_gather: List[asyncio.Task[ToolResult]] = []
    for t in message['tools']:
        assert t['call_type'] == 'function'
        async def _go(t: ToolCall) -> ToolResult:
            call_id = 'unknown'
            try:
                call_id = t['call_id']
                func = tools_map[t['function']['name']]
                args = t['function']['arguments']
                if inspect.iscoroutinefunction(func):
                    res = await func(**json.loads(args))
                else:
                    def _wrapped_sync() -> Any:
                        return func(**json.loads(args))
                    loop = asyncio.get_running_loop()
                    res = await loop.run_in_executor(None, _wrapped_sync)
                return {'call_id': call_id, 'result': res}
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                return {'call_id': call_id, 'result': error, 'is_error': True}
        to_gather.append(asyncio.create_task(_go(t)))
    return await asyncio.gather(*to_gather)


def build_tool_response_message(tool_results: List[ToolResult]) -> Message:
    return {
        'role': 'tool_res',
        'tools': tool_results,
    }
