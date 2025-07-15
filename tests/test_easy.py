import json

import pytest
from pydantic import BaseModel, Field

from lasagna.easy import (
    simple_ask,
    extract_prompt_text_from_pydantic_model,
    simple_ask_with_structured_output,
)
from lasagna.mock_provider import (
    MockProvider,
)
from lasagna.agent_util import bind_model


@pytest.mark.asyncio
async def test_easy_simple_ask_basic():
    result: str = await simple_ask(
        binder = bind_model(MockProvider, 'claude-3-5-sonnet-20240620'),
        system_prompt = 'Say hello world.',
        human_prompt = 'Hello, world!',
    )
    assert len(result) > 0, \
        f'result: {result}'
    assert isinstance(result, str), \
        f'result type: {type(result)}'


@pytest.mark.asyncio
async def test_easy_simple_ask_with_streaming():
    streaming_output: str = ""
    async def streaming_callback(text: str) -> None:
        nonlocal streaming_output
        streaming_output += text

    result: str = await simple_ask(
        binder = bind_model(MockProvider, 'claude-3-5-sonnet-20240620'),
        system_prompt = 'Say hello world.',
        human_prompt = 'Hello, world!',
        streaming_callback = streaming_callback,
    )
    assert len(streaming_output) > 0, \
        f'streaming_output: {streaming_output}'


@pytest.mark.asyncio
async def test_easy_simple_ask_with_tools():
    did_call_tool = False
    async def hammer_the_nail(type_of_hammer: str, type_of_nail: str) -> bool:
        """
        Try to hammer a nail.
        :param: type_of_hammer: str: The type of hammer to use.
        :param: type_of_nail: str: The type of nail to use.
        :return: bool: True if the nail is hammered, False otherwise.
        """
        nonlocal did_call_tool
        did_call_tool = True
        return True

    result: str = await simple_ask(
        binder = bind_model(MockProvider, 'claude-3-5-sonnet-20240620'),
        system_prompt = 'You are a nail hammerer.',
        human_prompt = 'Hammer the nail.',
        tools = [hammer_the_nail],
        force_tool = True,
        max_tool_iters = 1,
    )
    assert did_call_tool, \
        'did_call_tool: {did_call_tool}'


def test_easy_extract_prompt_text_from_pydantic_model_missing_description():
    class MyModel(BaseModel):
        name: str
        age: int

    with pytest.raises(ValueError):
        extract_prompt_text_from_pydantic_model(MyModel)


def test_easy_extract_prompt_text_from_pydantic_model():
    class MyModel(BaseModel):
        name: str = Field(description='The name of the person')
        age: int = Field(description='The age of the person')

    prompt_text = extract_prompt_text_from_pydantic_model(MyModel)
    expected_text = json.dumps(
        {
            'name': '(string) The name of the person',
            'age': '(integer) The age of the person',
        },
        indent=2,
    )
    assert prompt_text == expected_text, \
        f'prompt_text: {prompt_text}'


@pytest.mark.asyncio
async def test_easy_simple_ask_with_structured_output():
    class MyModel(BaseModel):
        name: str = Field(description='Make up a name of the person')
        age: int = Field(description='Make up an age of the person')

    result = await simple_ask_with_structured_output(
        binder = bind_model(MockProvider, 'claude-3-5-sonnet-20240620'),
        system_prompt = 'You are making stuff up.',
        human_prompt = 'Hello, world!',
        extraction_type = MyModel,
    )
    assert isinstance(result, MyModel), f"Expected MyModel type, got {type(result)}"
