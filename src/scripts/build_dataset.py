#!/usr/bin/env python3
"""
Universal dataset builder for multiple languages
Creates contrastive training datasets from Wikipedia and other sources
"""

import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import InputExample

LANGUAGE_CONFIGS = {
    "en": {
        "wikipedia": "20231101.en",
        "prompt": "Given a text, find semantically similar texts"
    },
    "de": {
        "wikipedia": "20231101.de", 
        "prompt": "Gegeben einen Text, finde semantisch ähnliche Texte"
    },
    "fr": {
        "wikipedia": "20231101.fr",
        "prompt": "Étant donné un texte, trouvez des textes sémantiquement similaires"
    },
    "es": {
        "wikipedia": "20231101.es",
        "prompt": "Dado un texto, encuentra textos semánticamente similares"
    },
    "it": {
        "wikipedia": "20231101.it",
        "prompt": "Dato un testo, trova testi semanticamente simili"
    },
    "pt": {
        "wikipedia": "20231101.pt",
        "prompt": "Dado um texto, encontre textos semanticamente semelhantes"
    },
    "nl": {
        "wikipedia": "20231101.nl",
        "prompt": "Gegeven een tekst, vind semantisch vergelijkbare teksten"
    },
    "pl": {
        "wikipedia": "20231101.pl",
        "prompt": "Mając dany tekst, znajdź semantycznie podobne teksty"
    },
    "ru": {
        "wikipedia": "20231101.ru",
        "prompt": "Дан текст, найдите семантически похожие тексты"
    },
    "ja": {
        "wikipedia": "20231101.ja",
        "prompt": "テキストが与えられたとき、意味的に類似したテキストを見つける"
    },
    "zh": {
        "wikipedia": "20231101.zh",
        "prompt": "给定一个文本，找到语义相似的文本"
    }
}

def build_simple_triplets(language="en", num_examples=1000, min_length=50):
    """Build simple triplets from Wikipedia"""
    
    if language not in LANGUAGE_CONFIGS:
        print(f"Warning: Language '{language}' not in predefined configs.")
        print(f"Available languages: {', '.join(LANGUAGE_CONFIGS.keys())}")
        print(f"Using generic config for '{language}'...")
        wiki_config = f"20231101.{language}"
    else:
        wiki_config = LANGUAGE_CONFIGS[language]["wikipedia"]
    
    print(f"Loading Wikipedia dataset for {language} ({wiki_config})...")
    
    try:
        # Use streaming to handle large datasets efficiently
        wiki = load_dataset("wikimedia/wikipedia", wiki_config, split="train", streaming=True)
        
        triplets = []
        examples_processed = 0
        
        for item in wiki:
            if examples_processed >= num_examples:
                break
                
            text = item['text']
            if len(text) < min_length:
                continue
                
            # Split into sentences (simple approach - works for most languages)
            # For better results, use language-specific sentence splitters
            sentences = []
            for delimiter in ['. ', '。', '। ', '။ ', '។ ']:  # Common sentence endings
                if delimiter in text:
                    sentences = text.split(delimiter)
                    break
            
            if not sentences:
                sentences = text.split('. ')
            
            # Clean sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) >= 3:
                # Create triplet: anchor, positive (nearby sentence), negative (distant sentence)
                idx = random.randint(0, len(sentences) - 3)
                anchor = sentences[idx]
                positive = sentences[idx + 1]  # Next sentence
                
                # Get negative from different part of text
                neg_idx = random.randint(max(0, idx - 5), min(len(sentences) - 1, idx + 5))
                while abs(neg_idx - idx) < 3 and len(sentences) > 5:
                    neg_idx = random.randint(0, len(sentences) - 1)
                negative = sentences[neg_idx]
                
                triplets.append({
                    'anchor': anchor,
                    'positive': positive, 
                    'negative': negative
                })
                
                examples_processed += 1
                if examples_processed % 100 == 0:
                    print(f"Processed {examples_processed} examples...")
        
        print(f"Created {len(triplets)} triplets from Wikipedia")
        return triplets
        
    except Exception as e:
        print(f"Error loading Wikipedia dataset: {e}")
        print("This might be due to:")
        print("- Invalid language code")
        print("- Network issues") 
        print("- Dataset not available for this language")
        return []

def load_specialized_datasets(language="en"):
    """Load language-specific specialized datasets if available"""
    datasets = []
    
    # Language-specific retrieval datasets
    retrieval_datasets = {
        "de": [
            ("mteb/germanquad-retrieval", "corpus"),
            ("mteb/GerDaLIRSmall", None)
        ],
        "fr": [
            ("unicamp-dl/mmarco", "french"),
        ],
        "en": [
            ("mteb/msmarco", None),
            ("sentence-transformers/nq", None)
        ]
    }
    
    if language in retrieval_datasets:
        for dataset_name, config in retrieval_datasets[language]:
            try:
                print(f"Loading {dataset_name}...")
                dataset = load_dataset(dataset_name, config, split="train")
                # Process based on dataset format
                # This is simplified - real implementation would handle each dataset's format
                print(f"Loaded {len(dataset)} examples from {dataset_name}")
            except Exception as e:
                print(f"Could not load {dataset_name}: {e}")
    
    return datasets

def create_llm2vec_format(triplets, language="en"):
    """Convert triplets to LLM2Vec training format"""
    
    # Get language-specific prompt
    prompt = LANGUAGE_CONFIGS.get(language, {}).get(
        "prompt", 
        f"Given a text in {language}, find semantically similar texts"
    )
    
    # Create output directory structure
    output_dir = Path("data/cache") / f"{language}-data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to E5-style format for LLM2Vec
    separator = "!@#$%^&*()"
    formatted_data = []
    
    for triplet in triplets:
        formatted_data.append({
            "query": f"{prompt}; {separator}{triplet['anchor']}",
            "positive": f"{separator}{triplet['positive']}",
            "negative": f"{separator}{triplet['negative']}"
        })
    
    # Save as JSONL
    output_file = output_dir / f"{language}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(formatted_data)} examples to {output_file}")
    
    # Also save raw format for reference
    raw_file = Path("data/datasets") / f"{language}_contrastive_dataset.json"
    with open(raw_file, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    
    # Create sample file
    sample_file = Path("data/datasets") / f"{language}_dataset_sample.txt"
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset samples for {language}:\n\n")
        for i, ex in enumerate(triplets[:10]):
            f.write(f"Example {i+1}:\n")
            f.write(f"Anchor: {ex['anchor'][:200]}...\n")
            f.write(f"Positive: {ex['positive'][:200]}...\n")
            f.write(f"Negative: {ex['negative'][:200]}...\n\n")
    
    return output_dir

def update_dataset_loader(language="en"):
    """Update the LLM2Vec dataset loader for the new language"""
    
    # Path to GermanData.py which we'll use as template
    template_path = Path("llm2vec/llm2vec/dataset/GermanData.py")
    if not template_path.exists():
        print("Warning: LLM2Vec not found. Dataset loader not updated.")
        return
    
    # For simplicity, we'll reuse the German loader with different paths
    # In production, you'd generate language-specific loaders
    
    print(f"\nTo use this dataset, update your training config:")
    print(f'  "dataset_name": "German",')
    print(f'  "dataset_file_path": "data/cache/{language}-data",')

def main():
    parser = argparse.ArgumentParser(
        description="Build contrastive training datasets for any language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # English dataset (default)
  python build_dataset.py
  
  # German dataset with 2000 examples
  python build_dataset.py --language de --num-examples 2000
  
  # French dataset
  python build_dataset.py --language fr
  
  # Multiple languages
  python build_dataset.py --language en,de,fr,es

Supported languages:
  en (English), de (German), fr (French), es (Spanish), it (Italian),
  pt (Portuguese), nl (Dutch), pl (Polish), ru (Russian), ja (Japanese), 
  zh (Chinese), and any other Wikipedia language code
        """
    )
    
    parser.add_argument("--language", type=str, default="en",
                       help="Language code(s), comma-separated for multiple")
    parser.add_argument("--num-examples", type=int, default=1000,
                       help="Number of examples to create per language")
    parser.add_argument("--min-length", type=int, default=50,
                       help="Minimum text length to consider")
    parser.add_argument("--include-specialized", action="store_true",
                       help="Include specialized datasets if available")
    
    args = parser.parse_args()
    
    # Process multiple languages if specified
    languages = [lang.strip() for lang in args.language.split(',')]
    
    for language in languages:
        print(f"\n{'='*60}")
        print(f"Building dataset for: {language}")
        print(f"{'='*60}")
        
        # Build Wikipedia-based triplets
        triplets = build_simple_triplets(
            language=language,
            num_examples=args.num_examples,
            min_length=args.min_length
        )
        
        if not triplets:
            print(f"Skipping {language} due to errors")
            continue
        
        # Add specialized datasets if requested
        if args.include_specialized:
            specialized = load_specialized_datasets(language)
            print(f"Added {len(specialized)} examples from specialized datasets")
            # triplets.extend(specialized)  # Would need format conversion
        
        # Convert to LLM2Vec format
        output_dir = create_llm2vec_format(triplets, language)
        
        # Update dataset loader
        update_dataset_loader(language)
        
        print(f"\n✓ Dataset for {language} complete!")
        print(f"  Examples: {len(triplets)}")
        print(f"  Output: {output_dir}")
    
    print("\n✓ All datasets built successfully!")
    print("\nNext steps:")
    print("1. To train a model with this dataset:")
    print(f'   python train.py "data/models/your-model-bi-init" --stage supervised')

if __name__ == "__main__":
    main()