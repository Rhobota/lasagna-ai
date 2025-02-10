from .agent_util import flat_messages
from .tools_util import extract_tool_result_as_sting

from .types import (
    EventCallback,
    EventPayload,
    BoundAgentCallable,
    AgentRun,
)

import os

from typing import Union, List

from colorama import just_fix_windows_console, Fore, Style


def make_tui_event_callback(
    truncate_past_chars: Union[int, None] = None,
    only_ai_messages: bool = False,
) -> EventCallback:
    async def tui_event_callback(event: EventPayload) -> None:
        just_fix_windows_console()
        if event[0] == 'ai' and event[1] == 'text_event':
            t = Style.RESET_ALL + event[2]
            print(t, end='', flush=True)
        elif only_ai_messages:
            pass
        elif event[0] == 'tool_call' and event[1] == 'text_event':
            assert isinstance(event[2], str)  # <-- mypy should *know* this, but it doesn't for some reason
            s = Fore.RED + event[2]
            print(s, end='', flush=True)
        elif event[0] == 'tool_res' and event[1] == 'tool_res_event':
            content = extract_tool_result_as_sting(event[2])
            if truncate_past_chars is None:
                env_truncate_past_chars = int(os.environ.get('LASAGNA_TUI_TOOL_RESULT_TRUNCATE', 50))
                content = _truncate_str(content, env_truncate_past_chars)
            elif truncate_past_chars > 0:
                content = _truncate_str(content, truncate_past_chars)
            r = Fore.BLUE + f" -> {content}"
            print(r)
        print(Style.RESET_ALL, end='', flush=True)

    return tui_event_callback


tui_event_callback = make_tui_event_callback()


async def tui_input_loop(
    bound_agent: BoundAgentCallable,
    system_prompt: Union[str, None] = None,
    truncate_past_chars: Union[int, None] = None,
    only_ai_messages: bool = False,
) -> None:
    just_fix_windows_console()
    event_callback = make_tui_event_callback(
        truncate_past_chars=truncate_past_chars,
        only_ai_messages=only_ai_messages,
    )
    prev_runs: List[AgentRun] = []
    if system_prompt is not None:
        prev_runs.append(flat_messages(
            'tui_input_loop',
            [
                {
                    'role': 'system',
                    'text': system_prompt,
                },
            ],
        ))
    try:
        while True:
            human_input = input(Fore.GREEN + Style.BRIGHT + '> ')
            print(Style.RESET_ALL, end='', flush=True)
            if not human_input:
                continue
            prev_runs.append(flat_messages(
                'tui_input_loop',
                [
                    {
                        'role': 'human',
                        'text': human_input,
                    },
                ],
            ))
            this_run = await bound_agent(event_callback, prev_runs)
            prev_runs.append(this_run)
            print(Style.RESET_ALL)
    except EOFError:
        # User hit ctrl-d
        pass
    except KeyboardInterrupt:
        # User hit ctrl-c
        pass
    finally:
        print(Style.RESET_ALL)


def _truncate_str(s: str, truncate_past_chars: int) -> str:
    if len(s) <= truncate_past_chars:
        return s
    truncated_s = s[:truncate_past_chars]
    n_truncated = len(s) - len(truncated_s)
    truncate_message = f' [... truncated {n_truncated} characters ...]'
    new_s = f'{truncated_s}{truncate_message}'
    if len(new_s) >= len(s):
        return s
    return new_s
