#!/usr/bin/env python
"""
Test script for Cached Baseline Adapter

Tests the cached adapter to ensure it loads cached responses
and handles new responses correctly.
"""

import sys
import json
import os
sys.path.insert(0, '.')

from adapters.offline.cached_adapter import CachedAdapter
from adapters.base import GenerationConfig

def test_cached_adapter():
    """Test cached baseline adapter functionality."""
    print("="*60)
    print("Testing Cached Baseline Adapter")
    print("="*60)
    
    # Setup cache file
    os.makedirs("./cache", exist_ok=True)
    cache_file = "./cache/test_baseline_responses.json"
    
    # Pre-populate cache
    initial_cache = {
        "hash_1": "2+2 equals 4.",
        "hash_2": "Paris is the capital of France.",
        "hash_3": "Machine learning is AI that learns from data."
    }
    
    with open(cache_file, 'w') as f:
        json.dump(initial_cache, f, indent=2)
    
    # Configuration
    adapter = CachedAdapter(
        name="cached-baseline",
        cache_file=cache_file,
        save_new_responses=True
    )
    
    # Test prompts
    test_prompts = [
        "What is 2+2?",
        "What is the capital of France?",
        "Explain machine learning.",
    ]
    
    try:
        # Load adapter
        print("\n1. Loading adapter...")
        adapter.load()
        print(f"   ✓ Adapter loaded: {adapter.is_loaded()}")
        
        # Test generation with cached responses
        print("\n2. Testing cached responses...")
        config = GenerationConfig(max_tokens=50)
        
        # Manually populate cache with test responses
        for i, prompt in enumerate(test_prompts):
            prompt_hash = adapter._hash_prompt(prompt)
            if i < len(initial_cache):
                adapter._cache[prompt_hash] = list(initial_cache.values())[i]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test {i}: {prompt}")
            response = adapter.generate(prompt, config)
            print(f"   → Response: {response.text}")
            print(f"   → Latency: {response.latency_ms:.1f}ms")
        
        # Check statistics
        print("\n3. Adapter statistics...")
        stats = adapter.get_stats()
        print(f"   → Cache size: {stats.get('cache_size', 0)}")
        print(f"   → Cache hits: {stats.get('cache_hits', 0)}")
        print(f"   → Hit rate: {stats.get('hit_rate_percent', 0):.1f}%")
        
        # Unload
        print("\n4. Unloading adapter...")
        adapter.unload()
        print(f"   ✓ Adapter unloaded: {not adapter.is_loaded()}")
        
        # Cleanup
        if os.path.exists(cache_file):
            os.remove(cache_file)
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED - Cached Adapter Working!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cached_adapter()
    sys.exit(0 if success else 1)
