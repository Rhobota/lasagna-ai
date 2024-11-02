from lasagna import (
    known_models,
    flat_messages,
    build_simple_agent,
    extract_last_message,
    noop_callback,
    Message,
)

from typing import List, Dict

import asyncio

from dotenv import load_dotenv; load_dotenv()


THE_AGENT = build_simple_agent(name='agent')


COMMITTEE_SPEC = [
    known_models.BIND_OPENAI_gpt_4o(),
    known_models.BIND_OPENAI_gpt_4o_mini(),

    known_models.BIND_ANTHROPIC_claude_35_sonnet(),
    known_models.BIND_ANTHROPIC_claude_3_opus(),
    known_models.BIND_ANTHROPIC_claude_3_haiku(),

    known_models.BIND_NVIDIA_meta_llama3_8b_instruct(),
    known_models.BIND_NVIDIA_meta_llama3_1_8b_instruct(),
    known_models.BIND_NVIDIA_meta_llama3_2_3b_instruct(),
    known_models.BIND_NVIDIA_mistralai_mistral_large(),
    known_models.BIND_NVIDIA_mistralai_mixtral_8x7b_instruct(),
    known_models.BIND_NVIDIA_google_recurrentgemma_2b(),
    known_models.BIND_NVIDIA_microsoft_phi_3_mini_128k_instruct(),
]


COMMITTEE_MODELS = [
    binder(THE_AGENT)
    for binder in COMMITTEE_SPEC
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
        last_message = extract_last_message(out, from_layered_agents=False)
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
