from lasagna import (     # <-- pip install -U lasagna-ai[openai,anthropic,bedrock]
    Model,
    EventCallback,
    AgentRun,
    recursive_extract_messages,
    flat_messages,
    known_models,
)

from lasagna.tui import tui_input_loop

from typing import List, Callable

import asyncio

import sympy as sp    # type: ignore

from dotenv import load_dotenv; load_dotenv()


def evaluate_math_expression(expression: str) -> float:
    """
    This tool evaluates a math expression and returns the result.
    Pass math expression as a string, for example:
     - "3 * 6 + 1"
     - "cos(2 * pi / 3) + log(8)"
     - "(4.5/2) + (6.3/1.2)"
     - ... etc
    :param: expression: str: the math expression to evaluate
    """
    expr = sp.sympify(expression)
    result = float(expr.evalf())
    return result


@known_models.openai_gpt_5_mini_binder
async def my_basic_agent(
    model: Model,
    event_callback: EventCallback,
    prev_runs: List[AgentRun],
) -> AgentRun:
    messages = recursive_extract_messages(prev_runs, from_tools=False, from_extraction=False)
    tools: List[Callable] = [
        evaluate_math_expression,
    ]
    new_messages = await model.run(event_callback, messages, tools=tools)
    this_run = flat_messages('my_basic_agent', new_messages)
    return this_run


async def main() -> None:
    system_prompt = """You are my math assistant. Your name is Matheo."""
    await tui_input_loop(my_basic_agent, system_prompt)


if __name__ == '__main__':
    asyncio.run(main())
