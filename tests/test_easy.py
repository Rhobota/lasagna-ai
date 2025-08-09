import pytest
from pydantic import BaseModel, Field

from lasagna import (
    make_model_binder,
    easy_ask,
    easy_extract,
)
from lasagna.mock_provider import (
    MockProvider,
)


@pytest.mark.asyncio
async def test_easy_ask():
    result: str = await easy_ask(
        binder = make_model_binder(MockProvider, 'fake_model'),
        prompt = 'Hello, world!',
        system_prompt = 'Say hello world.',
    )
    assert result == "\n".join([
        "Say hello world.",
        "Hello, world!",
        "model: fake_model",
    ])


@pytest.mark.asyncio
async def test_easy_extract():
    class MyModel(BaseModel):
        name: str = Field(description='Make up a name of the person')
        age: int = Field(description='Make up an age of the person')

    result = await easy_extract(
        binder = make_model_binder(MockProvider, 'fake_model', **dict(
            name = 'John Doe',
            age = 30,
        )),
        prompt = 'Hello, world!',
        system_prompt = 'You are making stuff up.',
        extraction_type = MyModel,
    )
    assert isinstance(result, MyModel), f"Expected MyModel type, got {type(result)}"
    assert result.name == 'John Doe'
    assert result.age == 30


@pytest.mark.asyncio
async def test_easy_ask_with_streaming():
    streaming_output: str = ""
    async def streaming_callback(text: str) -> None:
        nonlocal streaming_output
        streaming_output += text

    result: str = await easy_ask(
        binder = make_model_binder(MockProvider, 'fake_model'),
        prompt = 'Hello, world!',
        system_prompt = 'Say hello world.',
        streaming_callback = streaming_callback,
    )
    assert len(streaming_output) > 0, \
        f'streaming_output: {streaming_output}'

