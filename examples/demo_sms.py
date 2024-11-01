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


MODEL_BINDER = known_models.BIND_OPENAI_gpt_4o_mini()


# LLMs can hallucinate phone numbers, so you should be very cautious
# about letting them have access to this tool without guard rails!
# Below is a whitelist of phone numbers the agent is allowed to send to.
# Thus, if it hallucinates a phone number, we catch its mistake!
PHONE_NUMBER_WHITELIST: List[str] = [
    # PUT YOUR PHONE NUMBER(S) HERE, IN E.164 FORMAT
]


async def send_sms(to_phone_number: str, message_body: str) -> None:
    """
    Use this tool to send SMS message (aka, "text messages").
    :param: to_phone_number: str: the destination phone number to send to (in E.164 format; e.g. '+12223334444', where '+1' is the country code, '222' is the area code and '3334444' is the 7-digit phone number)
    :param: message_body: str: the content (aka, "body") of the SMS message
    """
    if to_phone_number not in PHONE_NUMBER_WHITELIST:
        raise PermissionError('you are not allowed to send SMS messages to that phone number')
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    account_auth_token = os.environ['TWILIO_AUTH_TOKEN']
    from_phone_number = os.environ['TWILIO_FROM_PHONE']
    payload = {
        'Body': message_body,
        'From': from_phone_number,
        'To':   to_phone_number,
    }
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    auth = aiohttp.BasicAuth(account_sid, account_auth_token)
    async with aiohttp.ClientSession(raise_for_status=True, auth=auth) as session:
        async with session.post(url, data=payload) as request:
            await request.json()


async def main() -> None:
    system_prompt = "You are grumpy."
    tools: List[Callable] = [
        send_sms,
    ]
    my_agent = build_simple_agent(name = 'agent', tools = tools)
    my_bound_agent = MODEL_BINDER(my_agent)
    await tui_input_loop(my_bound_agent, system_prompt)


if __name__ == '__main__':
    asyncio.run(main())
