#!/usr/bin/env python
"""
Download and test OpenAssistant model from HuggingFace
"""
import sys
sys.path.insert(0, '.')

print("="*60)
print("Downloading OpenAssistant Model")
print("="*60)

print("\n📦 Downloading OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5...")
print("   This may take several minutes...")

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    # Download tokenizer
    print("\n1. Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
    )
    print("   ✓ Tokenizer downloaded")
    
    # Download model (this will be large ~24GB)
    print("\n2. Downloading model (this will take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        torch_dtype="auto",
        low_cpu_mem_usage=True
    )
    print("   ✓ Model downloaded")
    
    # Quick test
    print("\n3. Testing model...")
    inputs = tokenizer("What is 2+2?", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Test response: {response}")
    
    print("\n" + "="*60)
    print("✓ OpenAssistant Model Ready!")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
