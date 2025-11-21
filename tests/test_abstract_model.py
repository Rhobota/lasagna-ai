import pytest

from lasagna.abstract_model import AbstractModel

from lasagna.agent_util import noop_callback


@pytest.mark.asyncio
async def test_abstract_model():
    model = AbstractModel(model = '')

    with pytest.raises(NotImplementedError):
        await model.run(
            event_callback = noop_callback,
            messages = [],
            tools = [],
        )

    with pytest.raises(NotImplementedError):
        await model.extract(
            event_callback = noop_callback,
            messages = [],
            extraction_type = int,
        )
