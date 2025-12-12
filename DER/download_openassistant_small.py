#!/usr/bin/env python
"""
Download OpenAssistant Falcon 7B model for 4-bit quantization
"""
import sys
import os

# Ensure we're in the right directory
if os.path.basename(os.getcwd()) != "DER":
    if os.path.exists("DER"):
        os.chdir("DER")

print("="*60)
print("Downloading OpenAssistant Falcon 7B Model")
print("="*60)

print("\n📦 Downloading OpenAssistant/falcon-7b-sft-mix-2000...")
print("   This is a 7B parameter model (~14GB in fp16).")
print("   We will use it in 4-bit mode (~5GB).")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_id = "OpenAssistant/falcon-7b-sft-mix-2000"
    
    # Download tokenizer
    print("\n1. Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("   ✓ Tokenizer downloaded")
    
    # Download model - we just download it here, usage will happen in training
    # We use from_pretrained to trigger the download to cache
    print("\n2. Downloading model to cache (this may take a few minutes)...")
    # We don't load into memory here to save time, just ensure files are present
    # Using low_cpu_mem_usage=True and just downloading config first to verify
    
    # Actually, let's just use snapshot_download from huggingface_hub to be cleaner
    try:
        from huggingface_hub import snapshot_download
        print("   Using huggingface_hub to download...")
        snapshot_download(repo_id=model_id)
        print("   ✓ Model downloaded to cache")
    except ImportError:
        print("   huggingface_hub not found, using transformers to download...")
        # Fallback to transformers load which caches it
        AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype="auto"
        )
        print("   ✓ Model downloaded to cache")
        
    print("\n" + "="*60)
    print("✓ OpenAssistant Model Ready for 4-bit Loading!")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Ensure sufficient disk space (~15GB)")
    sys.exit(1)
