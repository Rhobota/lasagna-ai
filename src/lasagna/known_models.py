from typing import List

from .types import ModelRecord


OPENAI_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'gpt-4o-mini-2024-07-18',
        'display_name': 'GPT-4o mini',
    },
    {
        'formal_name': 'gpt-4o-2024-08-06',
        'display_name': 'GPT-4o',
    },
    {
        'formal_name': 'gpt-4o-2024-05-13',
        'display_name': 'GPT-4o',
        'outdated': True,
    },
    {
        'formal_name': 'gpt-4-turbo-2024-04-09',
        'display_name': 'GPT-4',
        'outdated': True,
    },
    {
        'formal_name': 'gpt-3.5-turbo-0125',
        'display_name': 'GPT-3.5',
        'outdated': True,
    },
]


ANTHROPIC_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'claude-3-5-sonnet-20240620',
        'display_name': 'Claude3.5 Sonnet',
    },
    {
        'formal_name': 'claude-3-opus-20240229',
        'display_name': 'Claude3 Opus',
    },
    {
        'formal_name': 'claude-3-sonnet-20240229',
        'display_name': 'Claude3 Sonnet',
        'outdated': True,
    },
    {
        'formal_name': 'claude-3-haiku-20240307',
        'display_name': 'Claude3 Haiku',
    },
]


NVIDIA_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'meta/llama3-70b-instruct',
        'display_name': 'meta/llama3-70b-instruct',
    },
    {
        'formal_name': 'meta/llama3-8b-instruct',
        'display_name': 'meta/llama3-8b-instruct',
    },
    {
        'formal_name': 'mistralai/mistral-large',
        'display_name': 'mistralai/mistral-large',
    },
    {
        'formal_name': 'mistralai/codestral-22b-instruct-v0.1',
        'display_name': 'mistralai/codestral-22b-instruct-v0.1',
    },
    {
        'formal_name': 'mistralai/mixtral-8x22b-instruct-v0.1',
        'display_name': 'mistralai/mixtral-8x22b-instruct-v0.1',
    },
    {
        'formal_name': 'mistralai/mixtral-8x7b-instruct-v0.1',
        'display_name': 'mistralai/mixtral-8x7b-instruct-v0.1',
    },
    {
        'formal_name': 'google/gemma-7b',
        'display_name': 'google/gemma-7b',
    },
    {
        'formal_name': 'google/recurrentgemma-2b',
        'display_name': 'google/recurrentgemma-2b',
    },
    {
        'formal_name': 'microsoft/phi-3-mini-128k-instruct',
        'display_name': 'microsoft/phi-3-mini-128k-instruct',
    },
    {
        'formal_name': 'snowflake/arctic',
        'display_name': 'snowflake/arctic',
    },
]
