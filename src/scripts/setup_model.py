#!/usr/bin/env python3
"""
Universal setup script for any HuggingFace model with LLM2Vec
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from unittest.mock import patch

def setup_environment():
    """Set up environment variables"""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def mock_input_for_trust():
    """Mock function to auto-accept trust_remote_code prompts"""
    def _mock_input(prompt):
        if "Do you wish to run the custom code?" in prompt:
            print("Auto-accepting trust_remote_code...")
            return "y"
        return input(prompt)
    return _mock_input

def check_architecture_support(config_name):
    """Check if the model architecture is supported by LLM2Vec"""
    supported = {
        "LlamaConfig": "Llama-based model",
        "MistralConfig": "Mistral-based model", 
        "GemmaConfig": "Gemma-based model",
        "Qwen2Config": "Qwen2-based model",
        "SmolLM3Config": "SmolLM3-based model"
    }
    return config_name in supported, supported.get(config_name, "Unknown architecture")

def setup_bidirectional_model(model_id, trust_remote_code=False, output_dir=None):
    """Set up any HuggingFace model with bidirectional attention"""
    
    if output_dir is None:
        # Create a safe directory name from model ID
        safe_name = model_id.replace("/", "-").lower()
        output_dir = f"data/models/{safe_name}-bi-init"
    
    print(f"Setting up {model_id} with bidirectional attention...")
    print(f"Output directory: {output_dir}")
    
    # Import here to ensure environment is set up
    from transformers import AutoConfig, AutoTokenizer
    
    # Add current directory to Python path for llm2vec import
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    sys.path.insert(0, project_root)
    
    try:
        # First, try to load config to check architecture
        print("\nChecking model architecture...")
        if trust_remote_code:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        else:
            try:
                config = AutoConfig.from_pretrained(model_id)
            except ValueError as e:
                if "trust_remote_code" in str(e):
                    print("\nModel requires trust_remote_code. Retrying...")
                    trust_remote_code = True
                    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                else:
                    raise
        
        config_name = config.__class__.__name__
        print(f"Model config type: {config_name}")
        
        # Check if architecture is supported
        is_supported, arch_desc = check_architecture_support(config_name)
        if not is_supported:
            print(f"\n⚠️  Warning: {config_name} is not officially supported by LLM2Vec.")
            print("Supported architectures: Llama, Mistral, Gemma, Qwen2")
            print("The model will be loaded without bidirectional modifications.")
            response = input("\nContinue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("Setup cancelled.")
                return
        else:
            print(f"✓ Detected {arch_desc} - fully supported!")
        
        # Get model size info from config
        config_dict = config.to_dict()
        if 'num_parameters' in config_dict:
            param_count = config_dict['num_parameters']
        elif hasattr(config, 'num_parameters'):
            param_count = config.num_parameters
        else:
            # Estimate from hidden size and layers
            hidden_size = getattr(config, 'hidden_size', 0)
            num_layers = getattr(config, 'num_hidden_layers', 0)
            vocab_size = getattr(config, 'vocab_size', 0)
            param_count = (hidden_size * num_layers * 12 + vocab_size * hidden_size) if hidden_size > 0 else 0
        
        if param_count > 1e6:
            print(f"\nModel parameters: ~{param_count / 1e9:.1f}B")
        
        # Load the model with LLM2Vec
        print("\nLoading model with LLM2Vec...")
        
        # Handle trust_remote_code with mock input if needed
        if trust_remote_code:
            with patch('builtins.input', side_effect=mock_input_for_trust()):
                from llm2vec import LLM2Vec
                os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "True"
                model = LLM2Vec.from_pretrained(
                    model_id,
                    enable_bidirectional=is_supported,
                    device_map="auto",
                    torch_dtype="bfloat16",
                    pooling_mode="mean",
                    trust_remote_code=True
                )
        else:
            from llm2vec import LLM2Vec
            model = LLM2Vec.from_pretrained(
                model_id,
                enable_bidirectional=is_supported,
                device_map="auto", 
                torch_dtype="bfloat16",
                pooling_mode="mean"
            )
        
        # Save the model
        print(f"\nSaving model to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        model.save(output_dir)
        
        # Handle custom tokenizer files if they exist
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokenizer_files = ["tokenizer.py", "tokenizer_config.py", "gptx_tokenizer.py"]
        
        for file in tokenizer_files:
            cache_paths = [
                Path.home() / f".cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots",
                Path.home() / ".cache/huggingface/modules"
            ]
            
            for cache_path in cache_paths:
                if cache_path.exists():
                    for found_file in cache_path.rglob(file):
                        dest = Path(output_dir) / file
                        if not dest.exists():
                            shutil.copy2(found_file, dest)
                            print(f"Copied {file} to output directory")
                            break
        
        print("\n✓ Setup complete!")
        print(f"\nNext steps:")
        print(f"1. Build training dataset: python build_dataset.py --language <lang>")
        print(f"2. Train the model: python train.py \"{output_dir}\" --stage all")
        
        # Save setup info
        setup_info = {
            "base_model": model_id,
            "architecture": config_name,
            "bidirectional_enabled": is_supported,
            "trust_remote_code": trust_remote_code,
            "pooling_mode": "mean"
        }
        
        with open(Path(output_dir) / "setup_info.json", "w") as f:
            json.dump(setup_info, f, indent=2)
        
        return output_dir
        
    except Exception as e:
        print(f"\n❌ Error setting up model: {e}")
        print("\nCommon issues:")
        print("- Model ID not found (check spelling)")
        print("- Private model (login with: huggingface-cli login)")
        print("- Network issues (check connection)")
        print("- Insufficient memory (try smaller model)")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Set up any HuggingFace model for LLM2Vec embedding training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_model.py "HuggingFaceTB/SmolLM3-3B"
  python setup_model.py "meta-llama/Llama-3.2-1B" 
  python setup_model.py "openGPT-X/Teuken-7B-instruct-research-v0.4" --trust-remote-code
  python setup_model.py "mistralai/Mistral-7B-v0.1" --output "my-custom-dir"
        """
    )
    
    parser.add_argument("model_id", help="HuggingFace model ID (e.g., 'HuggingFaceTB/SmolLM3-3B')")
    parser.add_argument("--trust-remote-code", action="store_true", 
                       help="Trust remote code (required for some models like Teuken)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: data/models/{model-name}-bi-init)")
    
    args = parser.parse_args()
    
    setup_environment()
    setup_bidirectional_model(args.model_id, args.trust_remote_code, args.output)

if __name__ == "__main__":
    main()