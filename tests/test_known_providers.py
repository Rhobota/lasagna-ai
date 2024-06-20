from lasagna.known_providers import attempt_load_all_known_providers

from lasagna.registrar import (
    AGENTS,
    PROVIDERS,
)


def test_attempt_load_known_providers():
    AGENTS.clear()
    PROVIDERS.clear()
    attempt_load_all_known_providers()
    assert len(PROVIDERS) == 3
    assert len(AGENTS) == 0
