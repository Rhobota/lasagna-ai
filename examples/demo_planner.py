from lasagna import (
    make_model_binder,
    known_models,
    human_input,
    to_str,
)
from lasagna.tui import tui_event_callback
from lasagna.planner.agent import build_default_planning_agent
from lasagna.planner.debug_model import DebugModel

import os
import sys
import asyncio

from dotenv import load_dotenv; load_dotenv()


if os.environ.get('DEBUG', 'false').lower() in ['1', 'true', 'yes', 'y']:
    print('Using DEBUG Model')
    BINDER = make_model_binder(DebugModel, 'NA')
elif os.environ.get('ANTHROPIC_API_KEY'):
    print('Using Anthropic')
    BINDER = known_models.anthropic_claude_sonnet_4_5_binder
elif os.environ.get('OPENAI_API_KEY'):
    print('Using OpenAI')
    BINDER = known_models.openai_gpt_5_mini_binder
else:
    assert False, "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set! We need at least one to do this demo."


async def run_planner(prompt: str) -> None:
    # TODO Tools:
    #   - internet searching and link following
    #   - tools to access relevant contextual info:
    #       - user's location,
    #       - current time,
    #       - ask for clarification from the user,
    #       - etc.
    #   - access to file system (i.e. "agent memory")
    #   - access to a Python shell
    planning_agent = build_default_planning_agent(
        binder = BINDER,
    )
    run = await planning_agent(tui_event_callback, human_input(prompt))
    run_as_str = to_str(run)
    with open('planner_output.json', 'wt') as f:
        f.write(run_as_str)
    print(run_as_str)


def main() -> None:
    if sys.stdin.isatty():
        prompt = input('> ').strip()       # single-line input
    else:
        prompt = sys.stdin.read().strip()  # input until EOF
    asyncio.run(run_planner(prompt))


if __name__ == '__main__':
    main()
