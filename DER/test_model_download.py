#!/usr/bin/env python
"""Test model download with explicit safetensors parameter."""
import os
import sys

try:
    print("Importing transformers...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("SUCCESS: transformers imported")
    
    print("\nDownloading model with use_safetensors=True...")
    # Try forcing safetensors format
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m",
        use_safetensors=True
    )
    print("SUCCESS: model downloaded")
    
    print("\nDownloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    print("SUCCESS: tokenizer downloaded")
    
    print("\nSaving model with safetensors...")
    model_path = "./opt-125m"
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path, safe_serialization=True)
    print(f"SUCCESS: model saved to {model_path}")
    
    print("\nSaving tokenizer...")
    tokenizer.save_pretrained(model_path)
    print(f"SUCCESS: tokenizer saved to {model_path}")
    
    print("\nListing files in", model_path)
    for f in os.listdir(model_path):
        print(f"  - {f}")
    
    print("\nALL STEPS COMPLETED SUCCESSFULLY!")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
