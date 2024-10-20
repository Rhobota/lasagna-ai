from .registrar import register_provider

import logging

_LOG = logging.getLogger(__name__)


def attempt_load_all_known_providers() -> None:
    """
    Use this function when you want to populate the registrar
    at the start of the application. E.g. If you display a list of
    all known providers in the UI somewhere.
    """
    providers_to_try_to_load = [
        'openai',
        'anthropic',
        'nvidia',
    ]
    for provider in providers_to_try_to_load:
        try:
            attempt_load_known_providers(provider)
        except ImportError:
            _LOG.warning(f"failed to load provider: {provider}")


def attempt_load_known_providers(provider: str) -> None:
    """
    Attempt to load known providers at runtime, only when we *actually* need
    them.

    As the name implies, this is just an **attempt**.

    The `provider` string will be one of:
     1. known and loadable ✅, or...
     2. known but not loadable (probably due to a missing dependency) 😥, or...
     3. completely unknown 😥.

    Note: Normally I'd be sad by imports inside functions, but this is
          an acceptable use-case for doing so. We do *not* want to
          force people to install dependencies they don't need, thus
          we have to do some dynamic/runtime import-only-when-you-need-it
          behavior, which is what this function's job is. Moreover, that
          ugly behavior is restricted to *just this function* to keep the
          ugliness to a minimum.
    """
    if provider == 'openai':
        from .known_models import OPENAI_KNOWN_MODELS
        from .lasagna_openai import LasagnaOpenAI
        register_provider(
            key  = 'openai',
            name = 'OpenAI',
            factory = LasagnaOpenAI,
            models = OPENAI_KNOWN_MODELS,
        )

    elif provider == 'anthropic':
        from .known_models import ANTHROPIC_KNOWN_MODELS
        from .lasagna_anthropic import LasagnaAnthropic
        register_provider(
            key  = 'anthropic',
            name = 'Anthropic',
            factory = LasagnaAnthropic,
            models = ANTHROPIC_KNOWN_MODELS,
        )

    elif provider == 'nvidia':
        from .known_models import NVIDIA_KNOWN_MODELS
        from .lasagna_nvidia import LasagnaNVIDIA
        register_provider(
            key  = 'nvidia',
            name = 'NVIDIA',
            factory = LasagnaNVIDIA,
            models = NVIDIA_KNOWN_MODELS,
        )

    else:
        raise ValueError(f"unknown provider string: {provider}")
