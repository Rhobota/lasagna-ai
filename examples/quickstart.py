from lasagna import (
    bind_model,
    build_most_simple_agent,
)

from lasagna.tui import (
    tui_input_loop,
)

from typing import List, Callable

import asyncio

from dotenv import load_dotenv; load_dotenv()


MODEL_BINDER = bind_model('openai', 'gpt-3.5-turbo-0125')


async def main() -> None:
    system_prompt = "You are grumpy."
    tools: List[Callable] = []
    my_agent = build_most_simple_agent(tools)
    my_bound_agent = MODEL_BINDER(my_agent)
    await tui_input_loop(my_bound_agent, system_prompt)


if __name__ == '__main__':
    asyncio.run(main())
