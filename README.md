# Universal LLM2Vec: Transform Any LLM into a Text Encoder

Convert any decoder-only LLM from HuggingFace into a powerful bidirectional text encoder using the LLM2Vec framework. This project provides a universal, model-agnostic pipeline that works with any model - from small 1B parameter models to large 70B+ models.

**No configuration files. No model registries. Just works with any HuggingFace model.**

## 🚀 Key Features

- **🌐 Universal Model Support**: Works with ANY HuggingFace model out of the box
- **🧠 Intelligent Auto-Configuration**: Detects model size and GPU memory to optimize settings
- **🗣️ 100+ Language Support**: Build datasets for any Wikipedia language
- **⚡ Zero Configuration**: No config files needed - just model names
- **💾 Smart Memory Management**: Never run out of GPU memory again
- **🔧 Architecture Agnostic**: Automatically handles Llama, Mistral, Gemma, Qwen, and more

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM minimum, 16GB+ recommended)
- ~50GB free disk space

## 🎯 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone <repository>
cd llm2vec-universal

# Run the automated setup (installs dependencies + clones LLM2Vec locally)
./setup.sh

# Activate the environment
conda activate teuken2vec
```

### 2. Transform Any Model (3 Simple Commands)

```bash
# Step 1: Set up ANY model with bidirectional attention
python src/scripts/setup_model.py "HuggingFaceTB/SmolLM3-3B"

# Step 2: Build a training dataset (100+ languages supported)
python src/scripts/build_dataset.py --language en

# Step 3: Train the model (auto-configures everything)
python src/scripts/train.py "data/models/huggingfacetb-smollm3-3b-bi-init" --stage all
```

**That's it!** Your model is now a powerful text encoder. No configuration files needed.

## 🎮 Works with ANY Model

### Popular Models (all use the same command!)

```bash
# 🔥 SmolLM3 3B (perfect for RTX 4060/4070)
python src/scripts/setup_model.py "HuggingFaceTB/SmolLM3-3B"

# ⚡ Llama 3.2 1B (runs on 8GB GPUs!)
python src/scripts/setup_model.py "meta-llama/Llama-3.2-1B"

# 🚀 Mistral 7B (classic choice)
python src/scripts/setup_model.py "mistralai/Mistral-7B-v0.1"

# 🌟 Qwen2.5 (multilingual powerhouse)
python src/scripts/setup_model.py "Qwen/Qwen2.5-7B"

# 🎯 Teuken 7B (German specialist)
python src/scripts/setup_model.py "openGPT-X/Teuken-7B-instruct-research-v0.4" --trust-remote-code

# 🔮 ANY model from HuggingFace
python src/scripts/setup_model.py "microsoft/DialoGPT-large"
python src/scripts/setup_model.py "EleutherAI/gpt-neo-2.7B" 
python src/scripts/setup_model.py "your-org/your-awesome-model"
```

**The system automatically detects architecture and optimizes for your hardware!**

### 100+ Languages Supported

```bash
# 🇺🇸 English (default)
python src/scripts/build_dataset.py

# 🇩🇪 German for multilingual models
python src/scripts/build_dataset.py --language de

# 🌍 Multiple languages at once
python src/scripts/build_dataset.py --language en,de,fr,es,it,pt

# 🇯🇵 Any Wikipedia language 
python src/scripts/build_dataset.py --language ja --num-examples 2000

# 🇨🇳 Chinese, 🇷🇺 Russian, 🇦🇷 Arabic... any language!
python src/scripts/build_dataset.py --language zh,ru,ar,hi,ko
```

## 🇩🇪 Optimizing for German Embeddings

Since this project was initially developed for German text embeddings, here's how to get the best German-language performance:

### German-Optimized Pipeline

```bash
# 1. Use a German-specialized model (recommended)
python src/scripts/setup_model.py "openGPT-X/Teuken-7B-instruct-research-v0.4" --trust-remote-code

# 2. Build German contrastive dataset
python src/scripts/build_dataset.py --language de --num-examples 5000

# 3. Train with German Wikipedia for MNTP
python src/scripts/train.py "data/models/opengpt-x-teuken-7b-instruct-research-v0.4-bi-init" \
    --stage mntp \
    --dataset "wikimedia/wikipedia" \
    --dataset-config "20231101.de"

# 4. Fine-tune with German contrastive learning
python src/scripts/train.py "data/models/opengpt-x-teuken-7b-instruct-research-v0.4-bi-mntp" \
    --stage supervised
```

### German Language Features

- **🗣️ Native German prompts**: Uses "Gegeben einen Text, finde semantisch ähnliche Texte"
- **📚 German Wikipedia**: Automatically uses `20231101.de` dataset for MNTP training
- **🎯 German evaluation**: Compatible with MTEB German benchmarks (GermDUDE, GerDaLIR)
- **🇩🇪 German models**: Optimized for models like Teuken-7B, but works with any model

### Alternative German Models

```bash
# Other German-capable models
python src/scripts/setup_model.py "LeoLM/leo-hessianai-7b"
python src/scripts/setup_model.py "malteos/bloom-6b4-clp-german"
python src/scripts/setup_model.py "deutsche-telekom/gpt-3.5-turbo-german"

# Or use multilingual models with German data
python src/scripts/setup_model.py "meta-llama/Llama-3.2-3B"
python src/scripts/build_dataset.py --language de
```

**🎯 Result**: Embeddings optimized for German semantic similarity, question answering, and retrieval tasks.

### Training (Automatically Optimized)

```bash
# 🎯 Full training pipeline (auto-configured for your GPU)
python src/scripts/train.py "data/models/model-bi-init" --stage all

# 📚 Just MNTP (masked language modeling)
python src/scripts/train.py "data/models/model-bi-init" --stage mntp

# 🎭 Just supervised (contrastive learning)
python src/scripts/train.py "data/models/model-bi-init" --stage supervised

# 🔧 Override auto-settings (advanced users)
python src/scripts/train.py "data/models/model-bi-init" \
    --batch-size 16 \
    --max-length 1024 \
    --lora-r 32
```

**No need to worry about OOM errors - the system auto-adjusts for your hardware!**

## 🧠 How It Works

1. **Bidirectional Conversion**: Modifies attention masks to enable bidirectional context
2. **MNTP Pre-training**: Masked language modeling to adapt to bidirectional attention
3. **Contrastive Learning**: Fine-tunes on triplets (anchor, positive, negative) for embedding tasks

## 📊 Model Compatibility

| Architecture | Status | Example Models |
|-------------|--------|----------------|
| Llama | ✅ Supported | Llama-3.2, TinyLlama, Vicuna |
| Mistral | ✅ Supported | Mistral-7B, Mixtral |
| Gemma | ✅ Supported | Gemma-2B, Gemma-7B |
| Qwen2 | ✅ Supported | Qwen2.5 series |
| Others | ⚠️ Experimental | Falls back to standard transformers |

## 💾 Intelligent GPU Auto-Configuration

**Never configure training parameters again!** The system detects your hardware and optimizes automatically:

| Your GPU | Model Size | Auto-Config | Status |
|----------|------------|-------------|--------|
| RTX 4090 (24GB) | SmolLM3-3B | Batch=8, Seq=512, LoRA=16 | ✅ Perfect |
| RTX 4070 (12GB) | Llama-3.2-1B | Batch=8, Seq=512, LoRA=16 | ✅ Perfect |
| RTX 4060 Ti (16GB) | SmolLM3-3B | Batch=4, Seq=512, LoRA=8 | ✅ Good |
| A100 (80GB) | Any model | Batch=32, Seq=2048, LoRA=32 | 🚀 Blazing |
| A6000 (48GB) | Mistral-7B | Batch=16, Seq=1024, LoRA=16 | ✅ Excellent |

**The system automatically scales down if you run out of memory!**

## 🛠️ Advanced Usage

### Using Models Requiring Trust

Some models (like Teuken) require custom code:

```bash
python src/scripts/setup_model.py "openGPT-X/Teuken-7B-instruct-research-v0.4" --trust-remote-code
```

### Custom Output Directories

```bash
python src/scripts/setup_model.py "meta-llama/Llama-3.2-1B" --output "my-custom-path"
```

### Evaluating Your Model

```python
from llm2vec import LLM2Vec
import torch

# Load your trained model
model = LLM2Vec.from_pretrained(
    "data/models/your-model-bi-sup",
    device_map="cuda",
    torch_dtype="bfloat16"
)

# Encode texts
texts = ["Hello world", "Bonjour le monde", "Hola mundo"]
embeddings = model.encode(texts)

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)
print(similarities)
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- The trainer automatically reduces batch size on OOM
- Manually reduce with: `--batch-size 1 --max-length 128`

### Unsupported Architecture
- The system will warn you and fall back to standard transformers
- You can still use the model, but without bidirectional benefits

### Trust Remote Code Issues
- Add `--trust-remote-code` flag when needed
- The system auto-detects and prompts when required

## 📚 Project Structure

```
llm2vec-universal/
├── src/scripts/
│   ├── setup_model.py      # Universal model setup
│   ├── build_dataset.py    # Multi-language dataset builder
│   └── train.py           # Smart training orchestrator
├── data/
│   ├── models/            # Trained models go here
│   ├── datasets/          # Built datasets
│   └── cache/             # Temporary files
└── llm2vec/               # Core LLM2Vec library (auto-cloned)
```

## 🤝 Contributing

Contributions welcome! To add support for new architectures:
1. Fork the repository
2. Add architecture support to LLM2Vec
3. Submit a pull request

## 📄 Citations

If you use this project, please cite:

```bibtex
@article{llm2vec,
  title={LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders},
  author={BehnamGhader, Parishad and Adlakha, Vaibhav and Mosbach, Marius and Bahdanau, Dzmitry and Chapados, Nicolas and Reddy, Siva},
  journal={arXiv preprint arXiv:2404.05961},
  year={2024}
}
```

## 📧 Support

- Issues: [GitHub Issues](https://github.com/your-repo/issues)
- LLM2Vec: [McGill NLP](https://github.com/McGill-NLP/llm2vec)