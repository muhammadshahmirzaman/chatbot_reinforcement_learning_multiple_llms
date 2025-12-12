#!/usr/bin/env python
"""
Setup script to download and verify models for DER training.

This script downloads the models specified in the configuration
and verifies they work correctly.
"""

import os
import sys
import argparse
from pathlib import Path

# Add DER directory to path
sys.path.insert(0, str(Path(__file__).parent))

def download_model(model_name, use_safetensors=True):
    """Download a model from HuggingFace."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print('='*60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("  Loading model...")
        if use_safetensors:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_safetensors=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print(f"  ✓ {model_name} downloaded successfully!")
        print(f"    Model size: ~{model.num_parameters() / 1e6:.1f}M parameters")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download {model_name}: {e}")
        return False


def verify_models(config_path="./config/models_production.yaml"):
    """Verify all enabled models in the configuration."""
    print("\n" + "="*60)
    print("VERIFYING MODELS")
    print("="*60)
    
    try:
        from adapters.registry import ModelRegistry
        from adapters.base import GenerationConfig
        
        registry = ModelRegistry.from_config(config_path)
        models = registry.list_models()
        
        print(f"\nFound {len(models)} models in configuration")
        print(f"Models: {models}\n")
        
        config = GenerationConfig(max_tokens=10)
        test_prompt = "Hello, how are you?"
        
        for model_name in models:
            print(f"\nTesting {model_name}...")
            try:
                adapter = registry.get(model_name)
                adapter.load()
                
                response = adapter.generate(test_prompt, config)
                print(f"  ✓ Response: {response.text[:60]}...")
                print(f"    Latency: {response.latency_ms:.0f}ms")
                
                adapter.unload()
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print("\n" + "="*60)
        print("VERIFICATION COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser("DER Model Setup")
    parser.add_argument("--download", action="store_true", 
                       help="Download models from HuggingFace")
    parser.add_argument("--verify", action="store_true",
                       help="Verify models work correctly")
    parser.add_argument("--config", type=str, 
                       default="./config/models_production.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--models", nargs="+",
                       help="Specific models to download (space-separated)")
    
    args = parser.parse_args()
    
    if args.download:
        # Models to download (small, fast models by default)
        models_to_download = args.models or [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/phi-2",
            "microsoft/Phi-3-mini-4k-instruct",
        ]
        
        print("="*60)
        print("DOWNLOADING MODELS")
        print("="*60)
        print(f"\nWill download {len(models_to_download)} models:")
        for m in models_to_download:
            print(f"  - {m}")
        
        print("\nNote: First download may take several minutes...")
        print("Models will be cached in ~/.cache/huggingface/")
        
        for model_name in models_to_download:
            download_model(model_name)
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
    
    if args.verify:
        verify_models(args.config)
    
    if not args.download and not args.verify:
        print("Usage:")
        print("  python setup_production_models.py --download")
        print("  python setup_production_models.py --verify")
        print("  python setup_production_models.py --download --verify")
        print("\nOr download specific models:")
        print("  python setup_production_models.py --download --models TinyLlama/TinyLlama-1.1B-Chat-v1.0")


if __name__ == "__main__":
    main()
