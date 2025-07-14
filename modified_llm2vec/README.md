# Modified LLM2Vec

This directory contains the modified version of LLM2Vec with fixes for SmolLM3 support and transformers compatibility.

## Changes Made

### 1. SmolLM3 Support
- Added `SmolLM3Config` to the supported architectures in `llm2vec.py`
- SmolLM3 uses the existing `LlamaBiModel` implementation (compatible architecture)

### 2. Transformers 4.53.2 Compatibility Fixes
Fixed import issues in all bidirectional model implementations:

#### `bidirectional_mistral.py`
- Removed non-existent `MistralFlashAttention2` and `MistralSdpaAttention` imports
- Updated `MISTRAL_ATTENTION_CLASSES` to use only `ModifiedMistralAttention` for all attention implementations

#### `bidirectional_llama.py`
- Removed non-existent `LlamaFlashAttention2` and `LlamaSdpaAttention` imports
- Updated `LLAMA_ATTENTION_CLASSES` to use only `ModifiedLlamaAttention` for all attention implementations

#### `bidirectional_gemma.py`
- Removed non-existent `GemmaFlashAttention2` and `GemmaSdpaAttention` imports
- Updated `GEMMA_ATTENTION_CLASSES` to use only `ModifiedGemmaAttention` for all attention implementations

#### `bidirectional_qwen2.py`
- Removed non-existent `Qwen2FlashAttention2` and `Qwen2SdpaAttention` imports
- Updated `QWEN2_ATTENTION_CLASSES` to use only `ModifiedQwen2Attention` for all attention implementations

## Why These Changes Were Needed

In newer versions of transformers (4.53.2), the separate FlashAttention2 and SdpaAttention classes were consolidated into the main attention classes. The original LLM2Vec code was importing non-existent classes, causing import errors.

The fixes ensure that:
1. All attention implementations (`eager`, `flash_attention_2`, `sdpa`) use the same modified attention class
2. The `is_causal = False` modification is applied consistently across all attention implementations
3. SmolLM3 models can be used with the existing bidirectional conversion infrastructure

## Testing

All changes have been tested with:
- ✅ SmolLM3-3B model loading and encoding
- ✅ Existing model architectures (Llama, Mistral, Gemma, Qwen2)
- ✅ All attention implementations (eager, flash_attention_2, sdpa)

## Usage

To use the modified LLM2Vec:

```python
# Copy the modified files to your LLM2Vec installation
# or install LLM2Vec and replace the modified files

from llm2vec import LLM2Vec

# Now SmolLM3 is supported
model = LLM2Vec.from_pretrained(
    'HuggingFaceTB/SmolLM3-3B',
    enable_bidirectional=True,
    device_map="auto",
    torch_dtype="bfloat16",
    pooling_mode="mean"
)
```

## Original Source

These modifications are based on the original LLM2Vec repository: https://github.com/McGill-NLP/llm2vec