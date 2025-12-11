"""
Offline Model Adapter Test Script

This script tests the offline model adapters to verify they work correctly.
Run with: python test_offline_adapters.py

Usage:
    # Test with a small model (TinyLlama)
    python test_offline_adapters.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    
    # Test with a specific local model path
    python test_offline_adapters.py --model ./path/to/model
    
    # Test the cached adapter
    python test_offline_adapters.py --test-cached
"""

import sys
import argparse
import logging

sys.path.insert(0, '.')

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_transformers_adapter(model_path: str, device: str = "cuda"):
    """Test the TransformersAdapter with a real model."""
    print("\n" + "="*60)
    print("Testing TransformersAdapter")
    print("="*60)
    
    from adapters.offline.transformers_adapter import TransformersAdapter
    from adapters.base import GenerationConfig
    
    # Create adapter
    adapter = TransformersAdapter(
        name="test-transformers",
        model_name_or_path=model_path,
        device=device,
        torch_dtype="float16",
        estimated_memory_mb=3000
    )
    
    print(f"✓ Adapter created: {adapter}")
    print(f"  Execution mode: {adapter.execution_mode}")
    print(f"  Is loaded: {adapter.is_loaded()}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    adapter.load()
    print(f"✓ Model loaded successfully!")
    print(f"  Is loaded: {adapter.is_loaded()}")
    print(f"  Memory usage: {adapter.get_memory_usage()}")
    
    # Test generation
    print("\nTesting generation...")
    test_prompt = "What is machine learning? Answer briefly:"
    config = GenerationConfig(max_tokens=50, temperature=0.7)
    
    response = adapter.generate(test_prompt, config)
    print(f"✓ Generation successful!")
    print(f"  Prompt: {test_prompt}")
    print(f"  Response: {response.text}")
    print(f"  Latency: {response.latency_ms:.1f}ms")
    print(f"  Tokens: {response.tokens_used}")
    
    # Test offload
    print("\nTesting offload to CPU...")
    adapter.offload_to_cpu()
    print(f"✓ Offload successful!")
    
    # Unload
    print("\nUnloading model...")
    adapter.unload()
    print(f"✓ Model unloaded!")
    print(f"  Is loaded: {adapter.is_loaded()}")
    
    return True


def test_cached_adapter():
    """Test the CachedAdapter."""
    print("\n" + "="*60)
    print("Testing CachedAdapter")
    print("="*60)
    
    from adapters.offline.cached_adapter import CachedAdapter
    from adapters.base import GenerationConfig
    import tempfile
    import os
    
    # Create temporary cache file
    cache_file = os.path.join(tempfile.gettempdir(), "test_cache.json")
    
    adapter = CachedAdapter(
        name="test-cached",
        cache_file=cache_file,
        save_new_responses=True
    )
    
    print(f"✓ Adapter created: {adapter}")
    print(f"  Cache file: {cache_file}")
    
    # Load cache
    adapter.load()
    print(f"✓ Cache loaded!")
    print(f"  Is loaded: {adapter.is_loaded()}")
    
    # Test cache miss (no fallback)
    print("\nTesting cache miss...")
    response = adapter.generate("Test prompt 1")
    print(f"  Response: {response.text}")
    
    # Manually add to cache and test hit
    adapter._cache[adapter._hash_prompt("Hello world")] = "Cached response!"
    response = adapter.generate("Hello world")
    print(f"\n✓ Cache hit test:")
    print(f"  Prompt: Hello world")
    print(f"  Response: {response.text}")
    print(f"  Latency: {response.latency_ms}ms")
    
    # Get stats
    stats = adapter.get_stats()
    print(f"\n✓ Cache stats: {stats}")
    
    # Cleanup
    adapter.unload()
    if os.path.exists(cache_file):
        os.remove(cache_file)
    
    return True


def test_registry_with_offline():
    """Test the ModelRegistry with offline models enabled."""
    print("\n" + "="*60)
    print("Testing ModelRegistry with Offline Models")
    print("="*60)
    
    from adapters.registry import ModelRegistry
    
    # Load from config
    registry = ModelRegistry.from_config('./config/models.yaml')
    
    print(f"✓ Registry loaded!")
    print(f"  Total models: {registry.get_num_models()}")
    print(f"  Online models: {registry.list_online_models()}")
    print(f"  Offline models: {registry.list_offline_models()}")
    print(f"\n  Status: {registry.get_status()}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test offline model adapters")
    parser.add_argument("--model", type=str, 
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model to test with")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda, cpu)")
    parser.add_argument("--test-cached", action="store_true",
                        help="Test the cached adapter only")
    parser.add_argument("--test-registry", action="store_true",
                        help="Test the registry only")
    parser.add_argument("--skip-transformers", action="store_true",
                        help="Skip TransformersAdapter test (requires model download)")
    
    args = parser.parse_args()
    
    results = []
    
    # Test registry
    if args.test_registry:
        try:
            test_registry_with_offline()
            results.append(("Registry", True))
        except Exception as e:
            logger.error(f"Registry test failed: {e}")
            results.append(("Registry", False))
        return
    
    # Test cached adapter
    if args.test_cached:
        try:
            test_cached_adapter()
            results.append(("CachedAdapter", True))
        except Exception as e:
            logger.error(f"CachedAdapter test failed: {e}")
            results.append(("CachedAdapter", False))
        return
    
    # Full test suite
    print("\n" + "#"*60)
    print("# DER Offline Adapter Test Suite")
    print("#"*60)
    
    # Test cached adapter (always works)
    try:
        test_cached_adapter()
        results.append(("CachedAdapter", True))
    except Exception as e:
        logger.error(f"CachedAdapter test failed: {e}")
        results.append(("CachedAdapter", False))
    
    # Test transformers adapter
    if not args.skip_transformers:
        try:
            test_transformers_adapter(args.model, args.device)
            results.append(("TransformersAdapter", True))
        except Exception as e:
            logger.error(f"TransformersAdapter test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(("TransformersAdapter", False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
