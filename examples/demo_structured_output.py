from lasagna import (
    known_models,
    build_extraction_agent,
    flat_messages,
)

from lasagna.tui import tui_event_callback

from typing import List

from pydantic import BaseModel

import asyncio

from dotenv import load_dotenv; load_dotenv()


MODEL_BINDER = known_models.BIND_OPENAI_gpt_4o_mini()


PROMPT = """
Hey diddle diddle,
The cat and the fiddle,
The cow jumped over the moon;
The little dog laughed
To see such sport,
And the dish ran away with the spoon.
""".strip()


class LinguisticConstruction(BaseModel):
    subject: str
    verb: str
    object: str


class ExtractionModel(BaseModel):
    summary: str
    constructions: List[LinguisticConstruction]


async def main() -> None:
    my_bound_agent = MODEL_BINDER(
        build_extraction_agent(ExtractionModel),
    )
    prev_runs = [flat_messages([
        {
            'role': 'human',
            'text': PROMPT,
        },
    ])]

    agent_run = await my_bound_agent(tui_event_callback, prev_runs)
    print('DONE!')

    assert agent_run['type'] == 'extraction'
    result: ExtractionModel = agent_run['result']

    print(type(result))
    print(result.summary)

    for construction in result.constructions:
        print('   ', construction)


if __name__ == '__main__':
    asyncio.run(main())
