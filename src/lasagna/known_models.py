from typing import List

from .types import ModelRecord

from .agent_util import partial_bind_model


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

BIND_OPENAI_gpt_4o_mini  = partial_bind_model('openai', 'gpt-4o-mini-2024-07-18')
BIND_OPENAI_gpt_4o       = partial_bind_model('openai', 'gpt-4o-2024-08-06')
BIND_OPENAI_gpt_4_turbo  = partial_bind_model('openai', 'gpt-4-turbo-2024-04-09')
BIND_OPENAI_gpt_35_turbo = partial_bind_model('openai', 'gpt-3.5-turbo-0125')


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

BIND_ANTHROPIC_claude_35_sonnet = partial_bind_model('anthropic', 'claude-3-5-sonnet-20240620')
BIND_ANTHROPIC_claude_3_opus    = partial_bind_model('anthropic', 'claude-3-opus-20240229')
BIND_ANTHROPIC_claude_3_sonnet  = partial_bind_model('anthropic', 'claude-3-sonnet-20240229')
BIND_ANTHROPIC_claude_3_haiku   = partial_bind_model('anthropic', 'claude-3-haiku-20240307')


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
        'formal_name': 'meta/llama-3.1-8b-instruct',
        'display_name': 'meta/llama-3.1-8b-instruct',
    },
    {
        'formal_name': 'meta/llama-3.1-70b-instruct',
        'display_name': 'meta/llama-3.1-70b-instruct',
    },
    {
        'formal_name': 'meta/llama-3.1-405b-instruct',
        'display_name': 'meta/llama-3.1-405b-instruct',
    },
    {
        'formal_name': 'meta/llama-3.2-1b-instruct',
        'display_name': 'meta/llama-3.2-1b-instruct',
    },
    {
        'formal_name': 'meta/llama-3.2-3b-instruct',
        'display_name': 'meta/llama-3.2-3b-instruct',
    },
    {
        'formal_name': 'meta/llama-3.2-11b-vision-instruct',
        'display_name': 'meta/llama-3.2-11b-vision-instruct',
    },
    {
        'formal_name': 'meta/llama-3.2-90b-vision-instruct',
        'display_name': 'meta/llama-3.2-90b-vision-instruct',
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
]

BIND_NVIDIA_meta_llama3_70b_instruct           = partial_bind_model('nvidia', 'meta/llama3-70b-instruct')
BIND_NVIDIA_meta_llama3_8b_instruct            = partial_bind_model('nvidia', 'meta/llama3-8b-instruct')
BIND_NVIDIA_meta_llama3_1_8b_instruct          = partial_bind_model('nvidia', 'meta/llama-3.1-8b-instruct')
BIND_NVIDIA_meta_llama3_1_70b_instruct         = partial_bind_model('nvidia', 'meta/llama-3.1-70b-instruct')
BIND_NVIDIA_meta_llama3_1_405b_instruct        = partial_bind_model('nvidia', 'meta/llama-3.1-405b-instruct')
BIND_NVIDIA_meta_llama3_2_1b_instruct          = partial_bind_model('nvidia', 'meta/llama-3.2-1b-instruct')
BIND_NVIDIA_meta_llama3_2_3b_instruct          = partial_bind_model('nvidia', 'meta/llama-3.2-3b-instruct')
BIND_NVIDIA_meta_llama3_2_11b_vision_instruct  = partial_bind_model('nvidia', 'meta/llama-3.2-11b-vision-instruct')
BIND_NVIDIA_meta_llama3_2_90b_vision_instruct  = partial_bind_model('nvidia', 'meta/llama-3.2-90b-vision-instruct')

BIND_NVIDIA_mistralai_mistral_large            = partial_bind_model('nvidia', 'mistralai/mistral-large')
BIND_NVIDIA_mistralai_codestral_22b_instruct   = partial_bind_model('nvidia', 'mistralai/codestral-22b-instruct-v0.1')
BIND_NVIDIA_mistralai_mixtral_8x22b_instruct   = partial_bind_model('nvidia', 'mistralai/mixtral-8x22b-instruct-v0.1')
BIND_NVIDIA_mistralai_mixtral_8x7b_instruct    = partial_bind_model('nvidia', 'mistralai/mixtral-8x7b-instruct-v0.1')

BIND_NVIDIA_google_gemma_7b                    = partial_bind_model('nvidia', 'google/gemma-7b')
BIND_NVIDIA_google_recurrentgemma_2b           = partial_bind_model('nvidia', 'google/recurrentgemma-2b')

BIND_NVIDIA_microsoft_phi_3_mini_128k_instruct = partial_bind_model('nvidia', 'microsoft/phi-3-mini-128k-instruct')
