from lasagna import (
    bind_model,
    build_most_simple_agent,
    flat_messages,
)

from lasagna.tui import (
    tui_event_callback,
)

from lasagna.types import Message

from typing import List

from dotenv import load_dotenv; load_dotenv()

import asyncio


async def main() -> None:
    messages: List[Message] = [
        {
            'role': 'human',
            'text': "Describe this image:",
            'media': [
                {
                    'type': 'image',
                    'image': 'https://raw.githubusercontent.com/Rhobota/lasagna-ai/main/logos/lasagna-ai.png',
                },
            ],
        },
    ]
    bound_agent = bind_model('anthropic', 'claude-3-5-sonnet-20240620')(
        build_most_simple_agent(),
    )
    await bound_agent(tui_event_callback, [flat_messages(messages)])
    print()


if __name__ == '__main__':
    asyncio.run(main())
