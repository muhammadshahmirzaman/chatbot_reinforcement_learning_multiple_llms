#!/usr/bin/env python
"""
Test 4-bit loading of OpenAssistant Falcon 7B
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def test_loading():
    print("="*60)
    print("Testing 4-bit Loading of OpenAssistant Falcon 7B")
    print("="*60)
    
    model_id = "OpenAssistant/falcon-7b-sft-mix-2000"
    
    print("\n1. Checking 4-bit support...")
    if not torch.cuda.is_available():
        print("   ✗ CUDA not available, cannot test 4-bit loading")
        return False
        
    try:
        import bitsandbytes
        print(f"   ✓ bitsandbytes found: {bitsandbytes.__version__}")
    except ImportError:
        print("   ✗ bitsandbytes NOT found")
        return False

    print("\n2. Loading model in 4-bit mode...")
    try:
        # Define quantization config explicitly to verify
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("   ✓ Model loaded successfully!")
        print(f"   ✓ Memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
        
        # Test generation
        print("\n3. Testing generation...")
        input_text = "User: What is machine learning?\nAssistant:"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=50)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\nResponse:\n{response}")
        
    except Exception as e:
        print(f"\n✗ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    test_loading()
