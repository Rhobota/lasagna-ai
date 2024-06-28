from lasagna import (
    bind_model,
    build_most_simple_agent,
)

from lasagna.tui import (
    tui_input_loop,
)

from typing import List, Callable

from dotenv import load_dotenv; load_dotenv()

from twilio.rest import Client as TwilioClient   # type: ignore

import asyncio
import os


MODEL_BINDER = bind_model('openai', 'gpt-4o-2024-05-13')


def send_sms(to_phone_number: str, message_body: str) -> None:
    """
    Use this tool to send SMS message (aka, "text messages").
    :param: to_phone_number: str: the destination phone number to send to (in E.164 format; e.g. '+12223334444', where '+1' is the country code, '222' is the area code and '3334444' is the 7-digit phone number)
    :param: message_body: str: the content (aka, "body") of the SMS message
    """
    client = TwilioClient(
        os.environ['TWILIO_ACCOUNT_SID'],
        os.environ['TWILIO_AUTH_TOKEN'],
    )
    client.messages.create(
        body=message_body,
        from_=os.environ['TWILIO_FROM_PHONE'],
        to=to_phone_number,
    )


async def main() -> None:
    system_prompt = "You are grumpy."
    tools: List[Callable] = [
        send_sms,
    ]
    my_agent = build_most_simple_agent(tools)
    my_bound_agent = MODEL_BINDER(my_agent)
    await tui_input_loop(my_bound_agent, system_prompt)


if __name__ == '__main__':
    asyncio.run(main())
