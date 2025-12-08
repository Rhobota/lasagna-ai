from lasagna import (
    make_model_binder,
    known_models,
    flat_messages,
    human_input,
    to_str,
)
from lasagna import Model, EventCallback, AgentRun, Message
from lasagna.tui import tui_event_callback
from lasagna.planner01.agent import build_default_planning_agent
from lasagna.planner01.debug_model import DebugModel
from lasagna.async_util import async_throttle

import os
import sys
import json
import asyncio
import aiohttp
from zoneinfo import ZoneInfo
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from typing import Dict, List, Literal

from dotenv import load_dotenv; load_dotenv()

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


if os.environ.get('DEBUG', 'false').lower() in ['1', 'true', 'yes', 'y']:
    print('Using DEBUG Model')
    BINDER = make_model_binder(DebugModel, 'NA')
elif os.environ.get('ANTHROPIC_API_KEY'):
    print('Using Anthropic')
    BINDER = known_models.anthropic_claude_haiku_4_5_binder
elif os.environ.get('OPENAI_API_KEY'):
    print('Using OpenAI')
    BINDER = known_models.openai_gpt_5_mini_binder
else:
    assert False, "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set! We need at least one to do this demo."


class fetch_url:
    """
    This tool fetches the supplied `url` (via an HTTP GET request).
    This tool returns the content the page at the supplied URL (`url`).
    :param: url: str: the URL to fetch
    """
    async def __call__(
        self,
        model: Model,
        event_callback: EventCallback,
        prev_runs: List[AgentRun],
    ) -> AgentRun:
        # These asserts are part of the layered agent contract.
        # See tools_util._run_single_tool()
        assert len(prev_runs) == 1
        only_run = prev_runs[0]
        assert only_run['agent'] == '__upstream__'
        assert only_run['type'] == 'messages'
        messages = only_run['messages']
        assert len(messages) > 0
        last_message = messages[-1]
        assert last_message['role'] == 'tool_call'
        tools = last_message['tools']
        assert len(tools) == 1
        tool = tools[0]

        # This tool's signature:
        args = json.loads(tool['function']['arguments'])
        assert 'url' in args
        url = args['url']
        assert isinstance(url, str)

        # Grab the URL's GET content:
        headers = {'User-Agent': os.environ.get('UA_STRING', 'Lasagna Planner Demo')}
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as response:
                status = response.status
                content_type = response.content_type
                content = await response.text()

        # Use bs4 to strip away js and css (to reduce token count):
        if 'text/html' in content_type.lower():
            soup = BeautifulSoup(content, 'lxml')
            for el in soup(["script", "style"]):
                el.extract()
            content = soup.get_text()

        # Include the status code if it's not the expected 200:
        if status != 200:
            content = f"response HTTP status code: {status}\n\n{content}"

        # Prompt the downstream model:
        prompt_messages: List[Message] = [
            {
                'role': 'system',
                'text': 'Transcribe the important parts of this webpage into Markdown. Do not output a preamble or summary; output your Markdown translation *only*.',
            },
            {
                'role': 'human',
                'text': content,
            },
        ]
        new_messages = await model.run(
            event_callback,
            messages = prompt_messages,
            tools = [],
        )

        # Return the generated content:
        generated_messages = [*prompt_messages, *new_messages]
        return flat_messages('fetch_url', generated_messages)


@async_throttle(
    max_concurrent = int(os.environ.get('BRAVE_API_CNCR', 1)),
    max_per_second = 0.5 * float(os.environ.get('BRAVE_API_RPS', 1.0)),
)
async def _brave_search(
    search_type: Literal['news', 'web', 'images', 'vidoes'],
    q: str,
    n: int,
) -> str:
    headers: Dict[str, str] = {
        'x-subscription-token': os.environ['BRAVE_API_KEY'],
    }
    url = f'https://api.search.brave.com/res/v1/{search_type}/search'
    params: Dict[str, str] = {
        'q': q,
        'count': str(n),
        'safesearch': 'moderate',
        'spellcheck': 'false',
    }
    if search_type in ['web', 'news']:
        params['extra_snippets'] = 'true'
    if search_type == 'images':
        params['safesearch'] = 'strict'
    async with aiohttp.ClientSession(raise_for_status=True, headers=headers) as session:
        async with session.get(url, params=params) as response:
            response = await response.json()
    return json.dumps(response)


async def web_search(
    q: str,
    n: int,
) -> str:
    """
    Use this tool to perform a web search to retrieve documents relevant to
    the supplied `q` string.
    This tool returns a JSON-encoded payload of the web search results.
    This tool uses Brave's API for doing the web search, so you can use the
    search syntax for Brave's API.
    NOTE: The returned results will _not_ contain the full content of each
    retrieved document; rather, the returned results will contain small snippets
    from each document along with a URL to use to fetch the whole document.
    You can use the `fetch_url` to fetch those URLs, if you so choose.
    :param: q: str: the search string to send to the search engine
    :param: n: int: the number of results that should be returned by this search (must be a positive value, at most 20)
    """
    assert (0 < n <= 20), f"n cannot be {n}; it must be in the range (0, 20]"
    return await _brave_search('web', q, n)


async def news_search(
    q: str,
    n: int,
) -> str:
    """
    Use this tool to perform a news search to retrieve news events relevant to
    the supplied `q` string.
    This tool returns a JSON-encoded payload of the news search results.
    This tool uses Brave's API for doing the news search, so you can use the
    search syntax for Brave's API.
    NOTE: The returned results will _not_ contain the full content of each
    retrieved news story; rather, the returned results will contain small snippets
    from each news story along with a URL to use to fetch the source article.
    You can use the `fetch_url` to fetch those URLs, if you so choose.
    :param: q: str: the search string to send to the search engine
    :param: n: int: the number of results that should be returned by this search (must be a positive value, at most 100)
    """
    assert (0 < n <= 100), f"n cannot be {n}; it must be in the range (0, 100]"
    return await _brave_search('news', q, n)


def ask_user(question: str) -> str:
    """
    Use this tool to ask the user a clarifying question.
    For example, if the user's prompt was unclear, you can
    clarify it with them. Or if the user's prompt is clear,
    but it's missing a key piece of information, you can ask
    them for that piece of information.
    This tool returns the user's answer to your question.
    :param: question: str: the question you need to ask the user
    """
    return input(f'AI needs to know: {question}\n>>> ')


def user_location() -> str:
    """
    Use this tool to get the user's current location.
    Use this tool if you need to know the user's location to answer
    local-specific questions.
    """
    return os.environ.get('USER_LOCATION', '[user location is unknown]')


def current_time() -> str:
    """
    Use this tool to obtain the _current_ date and time.
    This is important if you are asked to do something relative
    to the current time, then you can use this tool to get the
    current date and time. Otherwise, you don't need this tool.
    The date and time will be returned as a string in the user's
    current timezone.
    """
    utc_now = datetime.now(timezone.utc)
    user_tz_name = os.environ.get('USER_TIMEZONE', 'US/Central')
    user_tz = ZoneInfo(user_tz_name)
    user_now = utc_now.astimezone(user_tz)
    user_now_str = user_now.isoformat()
    return '\n'.join([
        f'The user is in the timezone: {user_tz_name}',
        f'Now is {user_now_str}',
    ])


async def run_planner(prompt: str) -> None:
    # TODO Tools:
    #   - access to file system (i.e. "agent memory")
    #   - access to a Python shell
    planning_agent = build_default_planning_agent(
        binder = BINDER,
        tools = [
            fetch_url(),
            web_search,
            news_search,
            ask_user,
            user_location,
            current_time,
        ],
        max_tool_iters = 20,
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
