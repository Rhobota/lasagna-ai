from lasagna import (
    known_models,
    build_simple_agent,
)

from lasagna.tui import (
    tui_input_loop,
)

from typing import List, Callable

from dotenv import load_dotenv; load_dotenv()

import asyncio
import aiohttp
import os
from datetime import datetime


MODEL_BINDER = known_models.BIND_OPENAI_gpt_4o_mini()


async def perform_research(query: str) -> str:
    """
    Use this tool to do internet research, for both:
    1. to do research about things you don't know, and
    2. to get information about recent news or events.
    :param: query: str: the query string to use for the internet search
    """
    url = 'https://api.tavily.com/search'
    payload = {
      "api_key": os.environ['TAVILY_API_KEY'],
      "query": query,
      "search_depth": "basic",   # 'basic' or 'advanced'
      "include_answer": False,
      "include_images": False,
      "include_raw_content": False,
      "max_results": 5,
    }
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(url, json=payload) as request:
            response = await request.json()
    results = [
        f"Source: {obj['url']}\n\nContent: {obj['content']}"
        for obj in response['results']
    ]
    return '\n\n'.join(results)


async def main() -> None:
    today = datetime.now().strftime('%B %d, %Y')
    system_prompt = f"You are a grumpy research agent. Today is {today}."
    tools: List[Callable] = [
        perform_research,
    ]
    my_agent = build_simple_agent(name = 'agent', tools = tools)
    my_bound_agent = MODEL_BINDER(my_agent)
    await tui_input_loop(my_bound_agent, system_prompt)


if __name__ == '__main__':
    asyncio.run(main())
