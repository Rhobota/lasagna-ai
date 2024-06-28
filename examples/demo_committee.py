from lasagna import (
    bind_model,
    flat_messages,
    build_most_simple_agent,
    extract_last_message,
    noop_callback,
)

from lasagna.types import Message

from typing import List, Dict

import asyncio

from dotenv import load_dotenv; load_dotenv()


THE_AGENT = build_most_simple_agent()


COMMITTEE_SPEC = [
    ('openai',    'gpt-4o-2024-05-13'),
    ('openai',    'gpt-4-turbo-2024-04-09'),
    ('openai',    'gpt-3.5-turbo-0125'),
    ('anthropic', 'claude-3-5-sonnet-20240620'),
    ('anthropic', 'claude-3-opus-20240229'),
    ('anthropic', 'claude-3-sonnet-20240229'),
    ('anthropic', 'claude-3-haiku-20240307'),
    ('nvidia',    'meta/llama3-70b-instruct'),
    ('nvidia',    'meta/llama3-8b-instruct'),
    ('nvidia',    'mistralai/mistral-large'),
    ('nvidia',    'mistralai/mixtral-8x7b-instruct-v0.1'),
    ('nvidia',    'google/recurrentgemma-2b'),
    ('nvidia',    'microsoft/phi-3-mini-128k-instruct'),
    ('nvidia',    'snowflake/arctic'),
]


COMMITTEE_MODELS = [
    bind_model(*model)(THE_AGENT)
    for model in COMMITTEE_SPEC
]


async def vote_on_jokes(joke_a: str, joke_b: str) -> Dict[str, int]:
    messages: List[Message] = [
        {
            'role': 'system',
            'text': "The user will provider you with two jokes named `a` and `b`. You will output which is funniest. Output just a single character (`a` or `b`) to cast your vote.",
        },
        {
            'role': 'human',
            'text': f"a: {joke_a}\n\nb: {joke_b}\n\nRemember, output **only** a single token a or b without any preamble.",
        },
    ]
    tasks = [
        model(noop_callback, [flat_messages(messages)])
        for model in COMMITTEE_MODELS
    ]
    outputs = await asyncio.gather(*tasks)
    counter: Dict[str, int] = {}
    for spec, out in zip(COMMITTEE_SPEC, outputs):
        last_message = extract_last_message(out)
        assert last_message['role'] == 'ai'
        text = last_message['text']
        assert text
        text = text.strip()
        print(spec, text)
        if text not in counter:
            counter[text] = 0
        counter[text] += 1
    print('---------')
    return counter


async def main() -> None:
    #joke_a = "Why don't skeletons fight each other? They don't have the guts."
    #joke_a = "Why did the chicken cross the road? To get to the other side!"
    #joke_a = "I'm not funny."
    joke_a = "Why did the cat jump? Because cats like to jump..."
    joke_b = "Why do programmers prefer dark mode? Because light attracts bugs!"
    votes1 = await vote_on_jokes(joke_a, joke_b)
    votes2 = await vote_on_jokes(joke_b, joke_a)
    votes = {
        'a': (votes1.get('a', 0) + votes2.get('b', 0)),
        'b': (votes1.get('b', 0) + votes2.get('a', 0)),
    }
    print(votes)


if __name__ == '__main__':
    asyncio.run(main())
