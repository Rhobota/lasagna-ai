from lasagna import (
    known_models,
    build_simple_agent,
    flat_messages,
    Message,
)

from lasagna.tui import (
    tui_event_callback,
)

from typing import List

from dotenv import load_dotenv; load_dotenv()

import asyncio


MODEL_BINDER = known_models.BIND_ANTHROPIC_claude_35_sonnet()


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
    bound_agent = MODEL_BINDER(
        build_simple_agent(name='agent'),
    )
    await bound_agent(tui_event_callback, [flat_messages(messages)])
    print()


if __name__ == '__main__':
    asyncio.run(main())
