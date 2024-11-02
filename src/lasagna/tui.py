from .agent_util import flat_messages
from .tools_util import extract_tool_result_as_sting

from .types import (
    EventPayload,
    BoundAgentCallable,
    AgentRun,
)

from typing import Union, List

from colorama import just_fix_windows_console, Fore, Style


async def tui_event_callback(event: EventPayload) -> None:
    just_fix_windows_console()
    if event[0] == 'ai' and event[1] == 'text_event':
        t = Style.RESET_ALL + event[2]
        print(t, end='', flush=True)
    elif event[0] == 'tool_call' and event[1] == 'text_event':
        assert isinstance(event[2], str)  # <-- mypy should *know* this, but it doesn't for some reason
        s = Fore.RED + event[2]
        print(s, end='', flush=True)
    elif event[0] == 'tool_res' and event[1] == 'tool_res_event':
        content = extract_tool_result_as_sting(event[2])
        r = Fore.BLUE + f" -> {content}"
        print(r)
    print(Style.RESET_ALL, end='', flush=True)


async def tui_input_loop(
    bound_agent: BoundAgentCallable,
    system_prompt: Union[str, None] = None,
) -> None:
    just_fix_windows_console()
    prev_runs: List[AgentRun] = []
    if system_prompt is not None:
        prev_runs.append(flat_messages([{
            'role': 'system',
            'text': system_prompt,
        }]))
    try:
        while True:
            human_input = input(Fore.GREEN + Style.BRIGHT + '> ')
            print(Style.RESET_ALL, end='', flush=True)
            prev_runs.append(flat_messages([{
                'role': 'human',
                'text': human_input,
            }]))
            this_run = await bound_agent(tui_event_callback, prev_runs)
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
