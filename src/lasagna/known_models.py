from typing import List

from .types import ModelRecord

from .agent_util import partial_bind_model


OPENAI_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'gpt-4.1-2025-04-14',
        'display_name': 'GPT-4.1',
    },
    {
        'formal_name': 'gpt-4o-mini-2024-07-18',
        'display_name': 'GPT-4o mini',
    },
    {
        'formal_name': 'gpt-4o-2024-11-20',
        'display_name': 'GPT-4o',
    },
    {
        'formal_name': 'gpt-4o-2024-08-06',
        'display_name': 'GPT-4o',
        'outdated': True,
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

BIND_OPENAI_gpt_41       = partial_bind_model('openai', 'gpt-4.1-2025-04-14')
BIND_OPENAI_gpt_4o_mini  = partial_bind_model('openai', 'gpt-4o-mini-2024-07-18')
BIND_OPENAI_gpt_4o       = partial_bind_model('openai', 'gpt-4o-2024-11-20')


ANTHROPIC_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'claude-opus-4-20250514',
        'display_name': 'Claude Opus 4',
    },
    {
        'formal_name': 'claude-sonnet-4-20250514',
        'display_name': 'Claude Sonnet 4',
    },
    {
        'formal_name': 'claude-3-7-sonnet-20250219',
        'display_name': 'Claude Sonnet 3.7',
    },
    {
        'formal_name': 'claude-3-5-haiku-20241022',
        'display_name': 'Claude Haiku 3.5',
    },
    {
        'formal_name': 'claude-3-5-sonnet-20240620',
        'display_name': 'Claude Sonnet 3.5',
        'outdated': True,
    },
    {
        'formal_name': 'claude-3-opus-20240229',
        'display_name': 'Claude Opus 3',
        'outdated': True,
    },
    {
        'formal_name': 'claude-3-sonnet-20240229',
        'display_name': 'Claude Sonnet 3',
        'outdated': True,
    },
    {
        'formal_name': 'claude-3-haiku-20240307',
        'display_name': 'Claude Haiku 3',
        'outdated': True,
    },
]

BIND_ANTHROPIC_claude_opus_4    = partial_bind_model('anthropic', 'claude-opus-4-20250514')
BIND_ANTHROPIC_claude_sonnet_4  = partial_bind_model('anthropic', 'claude-sonnet-4-20250514')
BIND_ANTHROPIC_claude_sonnet_37 = partial_bind_model('anthropic', 'claude-3-7-sonnet-20250219')
BIND_ANTHROPIC_claude_haiku_35  = partial_bind_model('anthropic', 'claude-3-5-haiku-20241022')


OLLAMA_KNOWN_MODELS: List[ModelRecord] = [
    # We'll only list models here that (1) support tool-calling because
    # that's kind of the whole point of lasagna, and (2) we've tested
    # ourselves and pass our "vibe" test. Users are of course more than
    # welcome to use *other* Ollama models as they see fit.
    {
        'formal_name': 'llama3.2',
        'display_name': 'Meta Llama 3.2',
    },
    {
        'formal_name': 'mistral-small',
        'display_name': 'Mistral Small',
    },
    {
        'formal_name': 'mistral-large',
        'display_name': 'Mistral Large',
    },
]

BIND_OLLAMA_llama3_2 = partial_bind_model('ollama', 'llama3.2')
BIND_OLLAMA_mistral_small = partial_bind_model('ollama', 'mistral-small')
BIND_OLLAMA_mistral_large = partial_bind_model('ollama', 'mistral-large')


BEDROCK_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'us.anthropic.claude-3-haiku-20240307-v1:0',
        'display_name': 'Claude 3 Haiku',
    },
    {
        'formal_name': 'us.anthropic.claude-3-sonnet-20240229-v1:0',
        'display_name': 'Claude 3 Sonnet',
    },
    {
        'formal_name': 'us.anthropic.claude-3-opus-20240229-v1:0',
        'display_name': 'Claude 3 Opus',
    },
    {
        'formal_name': 'us.anthropic.claude-3-5-haiku-20241022-v1:0',
        'display_name': 'Claude 3.5 Haiku',
    },
    {
        'formal_name': 'us.anthropic.claude-3-5-sonnet-20240620-v1:0',
        'display_name': 'Claude 3.5 Sonnet',
    },
    {
        'formal_name': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'display_name': 'Claude 3.5 Sonnet v2',
    },
    {
        'formal_name': 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        'display_name': 'Claude 3.7 Sonnet',
    },
]

BIND_BEDROCK_Claude_3_Haiku       = partial_bind_model('bedrock', 'us.anthropic.claude-3-haiku-20240307-v1:0')
BIND_BEDROCK_Claude_3_Sonnet      = partial_bind_model('bedrock', 'us.anthropic.claude-3-sonnet-20240229-v1:0')
BIND_BEDROCK_Claude_3_Opus        = partial_bind_model('bedrock', 'us.anthropic.claude-3-opus-20240229-v1:0')
BIND_BEDROCK_Claude_3_5_Haiku     = partial_bind_model('bedrock', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
BIND_BEDROCK_Claude_3_5_Sonnet    = partial_bind_model('bedrock', 'us.anthropic.claude-3-5-sonnet-20240620-v1:0')
BIND_BEDROCK_Claude_3_5_Sonnet_v2 = partial_bind_model('bedrock', 'us.anthropic.claude-3-5-sonnet-20241022-v2:0')
BIND_BEDROCK_Claude_3_7_Sonnet    = partial_bind_model('bedrock', 'us.anthropic.claude-3-7-sonnet-20250219-v1:0')


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
