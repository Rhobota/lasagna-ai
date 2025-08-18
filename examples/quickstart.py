from lasagna import (     # <-- pip install -U lasagna-ai[openai,anthropic,bedrock]
    Model,
    EventCallback,
    AgentRun,
    recursive_extract_messages,
    flat_messages,
    known_models,
)

from lasagna.tui import tui_input_loop

from typing import List

import asyncio

from dotenv import load_dotenv; load_dotenv()


@known_models.openai_gpt_5_mini_binder
async def my_basic_agent(
    model: Model,
    event_callback: EventCallback,
    prev_runs: List[AgentRun],
) -> AgentRun:
    messages = recursive_extract_messages(prev_runs, from_tools=False, from_extraction=False)
    new_messages = await model.run(event_callback, messages, tools=[])
    this_run = flat_messages('my_basic_agent', new_messages)
    return this_run


async def main() -> None:
    system_prompt = """You are a grumpy assistant. Be helpful, brief, and grumpy. Your name is Grumble."""
    await tui_input_loop(my_basic_agent, system_prompt)


if __name__ == '__main__':
    asyncio.run(main())
