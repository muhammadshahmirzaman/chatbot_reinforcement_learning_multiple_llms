#!/usr/bin/env python
"""
Test script for TinyLlama 1.1B Adapter

Tests the TinyLlama transformer adapter to ensure it loads
and generates responses correctly.
"""

import sys
sys.path.insert(0, '.')

from adapters.offline.transformers_adapter import TransformersAdapter
from adapters.base import GenerationConfig

def test_tinyllama_adapter():
    """Test TinyLlama 1.1B adapter functionality."""
    print("="*60)
    print("Testing TinyLlama 1.1B Adapter")
    print("="*60)
    
    # Configuration
    adapter = TransformersAdapter(
        name="tinyllama-1.1b",
        model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="cuda:0",  # Use GPU if available
        dtype="float16",
        estimated_memory_mb=2500,
        use_safetensors=True
    )
    
    # Test prompts
    test_prompts = [
        "What is 2+2?",
        "What is the capital of France?",
        "Explain machine learning in one sentence.",
    ]
    
    try:
        # Load model
        print("\n1. Loading model...")
        adapter.load()
        print(f"   ✓ Model loaded: {adapter.is_loaded()}")
        
        # Test generation
        print("\n2. Testing generation...")
        config = GenerationConfig(
            max_tokens=50,
            temperature=0.7,
            do_sample=True
        )
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test {i}: {prompt}")
            response = adapter.generate(prompt, config)
            print(f"   → Response: {response.text[:100]}...")
            print(f"   → Latency: {response.latency_ms:.1f}ms")
        
        # Unload
        print("\n3. Unloading model...")
        adapter.unload()
        print(f"   ✓ Model unloaded: {not adapter.is_loaded()}")
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED - TinyLlama Adapter Working!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tinyllama_adapter()
    sys.exit(0 if success else 1)
