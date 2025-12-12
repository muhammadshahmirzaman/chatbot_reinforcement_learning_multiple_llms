#!/usr/bin/env python
"""
Test script for Phi-3 Mini Adapter

Tests the Microsoft Phi-3 Mini transformer adapter to ensure it loads
and generates responses correctly.
"""

import sys
sys.path.insert(0, '.')

from adapters.offline.transformers_adapter import TransformersAdapter
from adapters.base import GenerationConfig

def test_phi3mini_adapter():
    """Test Phi-3 Mini adapter functionality."""
    print("="*60)
    print("Testing Phi-3 Mini (3.8B) Adapter")
    print("="*60)
    
    # Configuration
    adapter = TransformersAdapter(
        name="phi-3-mini",
        model_name_or_path="microsoft/Phi-3-mini-4k-instruct",
        device="cuda:0",
        dtype="float16",
        estimated_memory_mb=8000,
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
        print("✓ ALL TESTS PASSED - Phi-3 Mini Adapter Working!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phi3mini_adapter()
    sys.exit(0 if success else 1)
