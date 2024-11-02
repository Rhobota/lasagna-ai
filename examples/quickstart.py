from lasagna import (
    known_models,
    build_simple_agent,
)

from lasagna.tui import (
    tui_input_loop,
)

from typing import List, Callable

import asyncio

from dotenv import load_dotenv; load_dotenv()


MODEL_BINDER = known_models.BIND_OPENAI_gpt_4o_mini()


async def main() -> None:
    system_prompt = "You are grumpy."
    tools: List[Callable] = []
    my_agent = build_simple_agent(name = 'agent', tools = tools)
    my_bound_agent = MODEL_BINDER(my_agent)
    await tui_input_loop(my_bound_agent, system_prompt)


if __name__ == '__main__':
    asyncio.run(main())
