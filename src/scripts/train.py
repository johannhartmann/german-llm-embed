#!/usr/bin/env python3
"""
Universal training script for LLM2Vec models
Automatically configures training parameters based on model size and available resources
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import torch
from transformers import AutoConfig

def get_gpu_memory():
    """Get available GPU memory in GB"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0

def get_model_size(model_path):
    """Estimate model size from config"""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Try to estimate parameter count
    hidden_size = config.get("hidden_size", 0)
    num_layers = config.get("num_hidden_layers", 0)
    vocab_size = config.get("vocab_size", 0)
    
    # Rough estimation (not exact but good enough for our purposes)
    params = (vocab_size * hidden_size + 
              num_layers * (4 * hidden_size * hidden_size + 2 * hidden_size))
    
    return params / 1e9  # Return in billions

def auto_configure_training(model_path, stage, gpu_memory=None):
    """Automatically configure training parameters based on model and GPU"""
    
    if gpu_memory is None:
        gpu_memory = get_gpu_memory()
    
    model_size = get_model_size(model_path)
    
    print(f"Detected GPU memory: {gpu_memory:.1f} GB")
    print(f"Estimated model size: {model_size:.1f}B parameters" if model_size else "Model size unknown")
    
    # Base configuration
    config = {
        "gradient_checkpointing": True,
        "torch_dtype": "bfloat16",
        "dataloader_num_workers": 4,
        "seed": 42,
        "report_to": "none"
    }
    
    # Configure based on model size and GPU memory
    if model_size is None:
        model_size = 7  # Default assumption
    
    # Memory-based configuration
    if gpu_memory >= 80:  # A100 80GB
        if model_size <= 3:
            config.update({
                "per_device_train_batch_size": 32,
                "gradient_accumulation_steps": 1,
                "max_seq_length": 2048,
                "lora_r": 32
            })
        elif model_size <= 7:
            config.update({
                "per_device_train_batch_size": 16,
                "gradient_accumulation_steps": 2,
                "max_seq_length": 1024,
                "lora_r": 32
            })
        else:
            config.update({
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "max_seq_length": 512,
                "lora_r": 16
            })
    elif gpu_memory >= 40:  # A100 40GB
        if model_size <= 3:
            config.update({
                "per_device_train_batch_size": 16,
                "gradient_accumulation_steps": 2,
                "max_seq_length": 1024,
                "lora_r": 16
            })
        elif model_size <= 7:
            config.update({
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "max_seq_length": 512,
                "lora_r": 16
            })
        else:
            config.update({
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 8,
                "max_seq_length": 512,
                "lora_r": 16
            })
    elif gpu_memory >= 24:  # RTX 3090/4090
        if model_size <= 3:
            config.update({
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "max_seq_length": 512,
                "lora_r": 16
            })
        elif model_size <= 7:
            config.update({
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 8,
                "max_seq_length": 512,
                "lora_r": 16
            })
        else:
            config.update({
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 16,
                "max_seq_length": 256,
                "lora_r": 8
            })
    elif gpu_memory >= 16:  # RTX 4060 Ti 16GB
        if model_size <= 3:
            config.update({
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 8,
                "max_seq_length": 512,
                "lora_r": 8
            })
        else:
            config.update({
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 32,
                "max_seq_length": 256,
                "lora_r": 8
            })
    else:  # <16GB
        config.update({
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "max_seq_length": 128,
            "lora_r": 4
        })
    
    # Stage-specific settings
    if stage == "mntp":
        config.update({
            "learning_rate": 1e-4,
            "num_train_epochs": 1,
            "warmup_steps": 100,
            "logging_steps": 50,
            "save_steps": 1000,
            "eval_steps": 500,
            "mlm_probability": 0.2,
            "mask_token_type": "blank",
            "stop_after_n_steps": 2000  # For quick testing
        })
    else:  # supervised
        config.update({
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "warmup_steps": 100,
            "logging_steps": 25,
            "save_steps": 200,
            "stop_after_n_steps": 500  # For quick testing
        })
    
    return config

def create_training_config(model_path, stage, output_path, dataset_config=None):
    """Create training configuration file"""
    
    # Get base configuration
    config = auto_configure_training(model_path, stage)
    
    # Model-specific configuration
    config["model_name_or_path"] = model_path
    
    # Set output directory
    model_name = Path(model_path).name
    if stage == "mntp":
        config["output_dir"] = f"data/models/{model_name.replace('-bi-init', '-bi-mntp')}"
        config["do_train"] = True
        config["do_eval"] = True
        config["eval_strategy"] = "steps"
        config["data_collator_type"] = "default"
        
        # Dataset configuration
        if dataset_config:
            config.update(dataset_config)
        else:
            # Auto-detect language from built datasets
            cache_dir = Path("data/cache")
            detected_lang = "en"  # default fallback
            
            if cache_dir.exists():
                # Look for language-specific cache directories
                for lang_dir in cache_dir.iterdir():
                    if lang_dir.is_dir() and lang_dir.name.endswith("-data"):
                        lang_code = lang_dir.name.replace("-data", "")
                        if len(lang_code) == 2:  # valid language code
                            detected_lang = lang_code
                            print(f"Detected language dataset: {detected_lang}")
                            break
            
            config["dataset_name"] = "wikimedia/wikipedia"
            config["dataset_config_name"] = f"20231101.{detected_lang}"
            
    else:  # supervised
        mntp_path = model_path.replace("-bi-init", "-bi-mntp")
        if Path(mntp_path).exists():
            config["model_name_or_path"] = mntp_path
        
        config["output_dir"] = f"data/models/{model_name.replace('-bi-init', '-bi-sup')}"
        config["bidirectional"] = True
        config["pooling_mode"] = "mean"
        
        # Use language-specific dataset if available
        language = dataset_config.get("language", "en") if dataset_config else "en"
        dataset_path = f"data/cache/{language}-data"
        
        if Path(dataset_path).exists():
            config["dataset_name"] = "German"  # Reusing German loader for all languages
            config["dataset_file_path"] = dataset_path
        else:
            # Fallback to German dataset if exists
            if Path("data/cache/german-data").exists():
                config["dataset_name"] = "German"
                config["dataset_file_path"] = "data/cache/german-data"
            else:
                print(f"Warning: No dataset found at {dataset_path}")
                print("Please run: python src/scripts/build_dataset.py --language {language}")
                
        config["remove_unused_columns"] = False
        config["do_train"] = True
        config["disable_tqdm"] = False
        config["save_only_model"] = True
    
    config["overwrite_output_dir"] = True
    config["per_device_eval_batch_size"] = config["per_device_train_batch_size"]
    
    # Save configuration
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config

def run_training(model_path, stage, config_path):
    """Run the actual training"""
    
    # Get project root
    # Get the project root (two levels up from src/scripts/)
    project_root = Path(__file__).parent.parent.parent
    
    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}/llm2vec:{env.get('PYTHONPATH', '')}"
    
    if stage == "mntp":
        script = project_root / "llm2vec/experiments/run_mntp.py"
    else:
        script = project_root / "llm2vec/experiments/run_supervised.py"
    
    # Check if script exists
    if not script.exists():
        print(f"\n❌ Error: Training script not found at {script}")
        print("Please ensure LLM2Vec is cloned: git clone https://github.com/McGill-NLP/llm2vec.git")
        sys.exit(1)
    
    # Run training
    cmd = [sys.executable, str(script), str(config_path)]
    print(f"\nRunning: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Universal training script for LLM2Vec models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all stages
  python train.py "data/models/smollm3-3b-bi-init" --stage all
  
  # Train only MNTP stage  
  python train.py "data/models/llama-3.2-1b-bi-init" --stage mntp
  
  # Train only supervised stage
  python train.py "data/models/teuken-7b-bi-init" --stage supervised
  
  # Custom configuration
  python train.py "data/models/my-model-bi-init" --stage all --batch-size 8 --max-length 1024
        """
    )
    
    parser.add_argument("model_path", help="Path to bidirectional model from setup_model.py")
    parser.add_argument("--stage", choices=["mntp", "supervised", "all"], default="all",
                       help="Training stage(s) to run")
    parser.add_argument("--dataset", type=str, help="Dataset for MNTP (e.g., 'wikimedia/wikipedia')")
    parser.add_argument("--dataset-config", type=str, help="Dataset config (e.g., '20231101.de')")
    parser.add_argument("--batch-size", type=int, help="Override auto-configured batch size")
    parser.add_argument("--max-length", type=int, help="Override max sequence length")
    parser.add_argument("--lora-r", type=int, help="Override LoRA rank")
    parser.add_argument("--gpu-memory", type=float, help="Override GPU memory detection (in GB)")
    
    args = parser.parse_args()
    
    # Check model path exists
    if not Path(args.model_path).exists():
        print(f"❌ Error: Model path '{args.model_path}' does not exist")
        print("Run setup_model.py first to create a bidirectional model")
        sys.exit(1)
    
    stages = ["mntp", "supervised"] if args.stage == "all" else [args.stage]
    
    for stage in stages:
        print(f"\n{'='*60}")
        print(f"Training stage: {stage.upper()}")
        print(f"{'='*60}")
        
        # Create configuration
        config_path = Path("temp_train_config.json")
        
        dataset_config = {}
        if args.dataset:
            dataset_config["dataset_name"] = args.dataset
        if args.dataset_config:
            dataset_config["dataset_config_name"] = args.dataset_config
            
        config = create_training_config(
            args.model_path, 
            stage, 
            config_path,
            dataset_config
        )
        
        # Apply overrides
        if args.batch_size:
            config["per_device_train_batch_size"] = args.batch_size
            config["per_device_eval_batch_size"] = args.batch_size
        if args.max_length:
            config["max_seq_length"] = args.max_length
        if args.lora_r:
            config["lora_r"] = args.lora_r
            
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nTraining configuration:")
        print(f"  Batch size: {config['per_device_train_batch_size']}")
        print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
        print(f"  Effective batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
        print(f"  Max sequence length: {config['max_seq_length']}")
        print(f"  LoRA rank: {config['lora_r']}")
        print(f"  Output: {config['output_dir']}")
        
        # Run training
        run_training(args.model_path, stage, config_path)
        
        # Clean up
        if config_path.exists():
            config_path.unlink()
    
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()