from typing import List

from .types import ModelRecord

from .agent_util import make_model_binder


OPENAI_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'gpt-5-2025-08-07',
        'display_name': 'GPT-5',
    },
    {
        'formal_name': 'gpt-5-mini-2025-08-07',
        'display_name': 'GPT-5 mini',
    },
    {
        'formal_name': 'gpt-5-nano-2025-08-07',
        'display_name': 'GPT-5 nano',
    },
    {
        'formal_name': 'gpt-4.1-2025-04-14',
        'display_name': 'GPT-4.1',
    },
    {
        'formal_name': 'gpt-4o-mini-2024-07-18',
        'display_name': 'GPT-4o mini',
        'outdated': True,
    },
    {
        'formal_name': 'gpt-4o-2024-11-20',
        'display_name': 'GPT-4o',
        'outdated': True,
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

openai_gpt_5_binder        = make_model_binder('openai', 'gpt-5-2025-08-07')
openai_gpt_5_mini_binder   = make_model_binder('openai', 'gpt-5-mini-2025-08-07')
openai_gpt_5_nano_binder   = make_model_binder('openai', 'gpt-5-nano-2025-08-07')
openai_gpt_4_1_binder      = make_model_binder('openai', 'gpt-4.1-2025-04-14')


ANTHROPIC_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'claude-opus-4-1-20250805',
        'display_name': 'Claude Opus 4.1',
    },
    {
        'formal_name': 'claude-sonnet-4-5-20250929',
        'display_name': 'Claude Sonnet 4.5',
    },
    {
        'formal_name': 'claude-haiku-4-5-20251001',
        'display_name': 'Claude Haiku 4.5',
    },
    {
        'formal_name': 'claude-3-5-haiku-20241022',
        'display_name': 'Claude Haiku 3.5',
        'outdated': True,
    },
    {
        'formal_name': 'claude-opus-4-20250514',
        'display_name': 'Claude Opus 4',
        'outdated': True,
    },
    {
        'formal_name': 'claude-sonnet-4-20250514',
        'display_name': 'Claude Sonnet 4',
        'outdated': True,
    },
    {
        'formal_name': 'claude-3-7-sonnet-20250219',
        'display_name': 'Claude Sonnet 3.7',
        'outdated': True,
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

anthropic_claude_opus_4_1_binder  = make_model_binder('anthropic', 'claude-opus-4-1-20250805')
anthropic_claude_sonnet_4_5_binder  = make_model_binder('anthropic', 'claude-sonnet-4-5-20250929')
anthropic_claude_haiku_4_5_binder = make_model_binder('anthropic', 'claude-haiku-4-5-20251001')


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

ollama_llama3_2_binder = make_model_binder('ollama', 'llama3.2')
ollama_mistral_small_binder = make_model_binder('ollama', 'mistral-small')
ollama_mistral_large_binder = make_model_binder('ollama', 'mistral-large')


BEDROCK_KNOWN_MODELS: List[ModelRecord] = [
    {
        'formal_name': 'us.anthropic.claude-opus-4-1-20250805-v1:0',
        'display_name': 'Claude Opus 4.1',
    },
    {
        'formal_name': 'us.anthropic.claude-sonnet-4-20250514-v1:0',
        'display_name': 'Claude Sonnet 4',
    },
    {
        'formal_name': 'us.anthropic.claude-3-5-haiku-20241022-v1:0',
        'display_name': 'Claude Haiku 3.5',
    },
    {
        'formal_name': 'us.anthropic.claude-opus-4-20250514-v1:0',
        'display_name': 'Claude Opus 4',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        'display_name': 'Claude Sonnet 3.7',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-5-sonnet-20240620-v1:0',
        'display_name': 'Claude Sonnet 3.5',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'display_name': 'Claude Sonnet 3.5 (v2)',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-opus-20240229-v1:0',
        'display_name': 'Claude Opus 3',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-sonnet-20240229-v1:0',
        'display_name': 'Claude Sonnet 3',
        'outdated': True,
    },
    {
        'formal_name': 'us.anthropic.claude-3-haiku-20240307-v1:0',
        'display_name': 'Claude Haiku 3',
        'outdated': True,
    },
]

bedrock_claude_opus_4_1_binder = make_model_binder('bedrock', 'us.anthropic.claude-opus-4-1-20250805-v1:0')
bedrock_claude_sonnet_4_binder = make_model_binder('bedrock', 'us.anthropic.claude-sonnet-4-20250514-v1:0')
bedrock_claude_haiku_3_5_binder = make_model_binder('bedrock', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')


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

nvidia_meta_llama3_70b_instruct_binder           = make_model_binder('nvidia', 'meta/llama3-70b-instruct')
nvidia_meta_llama3_8b_instruct_binder            = make_model_binder('nvidia', 'meta/llama3-8b-instruct')
nvidia_meta_llama3_1_8b_instruct_binder          = make_model_binder('nvidia', 'meta/llama-3.1-8b-instruct')
nvidia_meta_llama3_1_70b_instruct_binder         = make_model_binder('nvidia', 'meta/llama-3.1-70b-instruct')
nvidia_meta_llama3_1_405b_instruct_binder        = make_model_binder('nvidia', 'meta/llama-3.1-405b-instruct')
nvidia_meta_llama3_2_1b_instruct_binder          = make_model_binder('nvidia', 'meta/llama-3.2-1b-instruct')
nvidia_meta_llama3_2_3b_instruct_binder          = make_model_binder('nvidia', 'meta/llama-3.2-3b-instruct')
nvidia_meta_llama3_2_11b_vision_instruct_binder  = make_model_binder('nvidia', 'meta/llama-3.2-11b-vision-instruct')
nvidia_meta_llama3_2_90b_vision_instruct_binder  = make_model_binder('nvidia', 'meta/llama-3.2-90b-vision-instruct')

nvidia_mistralai_mistral_large_binder            = make_model_binder('nvidia', 'mistralai/mistral-large')
nvidia_mistralai_codestral_22b_instruct_binder   = make_model_binder('nvidia', 'mistralai/codestral-22b-instruct-v0.1')
nvidia_mistralai_mixtral_8x22b_instruct_binder   = make_model_binder('nvidia', 'mistralai/mixtral-8x22b-instruct-v0.1')
nvidia_mistralai_mixtral_8x7b_instruct_binder    = make_model_binder('nvidia', 'mistralai/mixtral-8x7b-instruct-v0.1')

nvidia_google_gemma_7b_binder                    = make_model_binder('nvidia', 'google/gemma-7b')
nvidia_google_recurrentgemma_2b_binder           = make_model_binder('nvidia', 'google/recurrentgemma-2b')

nvidia_microsoft_phi_3_mini_128k_instruct_binder = make_model_binder('nvidia', 'microsoft/phi-3-mini-128k-instruct')
