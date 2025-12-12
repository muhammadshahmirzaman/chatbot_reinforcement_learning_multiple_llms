#!/usr/bin/env python
"""
Test script for OPT-125M Adapter

Tests the OPT-125M transformer adapter to ensure it loads
and generates responses correctly.
"""

import sys
sys.path.insert(0, '.')

from adapters.offline.transformers_adapter import TransformersAdapter
from adapters.base import GenerationConfig

def test_opt125m_adapter():
    """Test OPT-125M adapter functionality."""
    print("="*60)
    print("Testing OPT-125M Adapter")
    print("="*60)
    
    # Configuration
    adapter = TransformersAdapter(
        name="opt-125m",
        model_name_or_path="./opt-125m",
        device="cpu",  # Use CPU for compatibility
        torch_dtype="float32",
        estimated_memory_mb=500,
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
            max_tokens=30,
            temperature=0.7,
            do_sample=True
        )
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test {i}: {prompt}")
            response = adapter.generate(prompt, config)
            print(f"   → Response: {response.text[:80]}...")
            print(f"   → Latency: {response.latency_ms:.1f}ms")
            print(f"   → Tokens: ~{len(response.text.split())}")
        
        # Unload
        print("\n3. Unloading model...")
        adapter.unload()
        print(f"   ✓ Model unloaded: {not adapter.is_loaded()}")
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED - OPT-125M Adapter Working!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_opt125m_adapter()
    sys.exit(0 if success else 1)
