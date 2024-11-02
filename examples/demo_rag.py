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

import svs


MODEL_BINDER = known_models.BIND_OPENAI_gpt_4o_mini()


async def make_rag_tool() -> Callable:
    kb = svs.AsyncKB('https://github.com/Rhobota/svs/raw/main/examples/dad_jokes/dad_jokes.sqlite.gz')

    # It's best to tell the KB to start loading in the background *immediately*.
    # Otherwise it will lazy-load (i.e. wait to start loading until you query it).
    # We'd rather not have our first user wait, so we'll start loading it *now*.
    asyncio.create_task(kb.load())

    async def query_dad_jokes(query: str) -> str:
        """
        This tool retrieves "Dad Jokes" from a knowledge base. You pass in a
        search string (`query`) to describe the desired topic, and this tool
        retrieves 5 jokes about that topic. For example, if you want jokes about
        cats, you may pass in the string "cats" as the query.
        :param: query: str: the search string to use to retrieve 5 jokes
        """
        results = await kb.retrieve(query, n = 5)
        return '\n\n'.join([
            f"Joke #{i+1}: {result['doc']['text']}"
            for i, result in enumerate(results)
        ])

    return query_dad_jokes


SYSTEM_PROMPT = ' '.join("""
You are an assistant whose primary job is to tell the user "Dad Jokes", when
asked. A "Dad Joke" is a silly joke that dads are notorious for telling
their kids, much to their kids' dismay (because the jokes are often cheesy
and low-quality). You use the RAG-style tool named `query_dad_jokes` to
retrieve jokes from a curated knowledge base. That tool returns 5 jokes;
when the user asks you for a joke, you'll retrieve those 5 jokes then pick
only the best **one** to give back to the user.
""".strip().splitlines())


async def main() -> None:
    tools: List[Callable] = [
        await make_rag_tool(),
    ]
    my_agent = build_simple_agent(name = 'agent', tools = tools)
    my_bound_agent = MODEL_BINDER(my_agent)
    await tui_input_loop(my_bound_agent, SYSTEM_PROMPT)


if __name__ == '__main__':
    asyncio.run(main())
