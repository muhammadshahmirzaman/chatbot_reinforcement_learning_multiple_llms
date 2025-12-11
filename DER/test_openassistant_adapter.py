"""
OpenAssistant Adapter Test Script

This script tests the OpenAssistantAdapter with automatic model setup.
Just run: python test_openassistant_adapter.py

The script will:
1. Auto-download a test model if not present
2. Run all adapter tests (create, load, generate, offload, unload)
3. Report results
"""

import sys
import os
sys.path.insert(0, '.')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default test model path
DEFAULT_MODEL_PATH = "./opt-125m"
DEFAULT_MODEL_HF = "facebook/opt-125m"


def ensure_model_exists(model_path: str = DEFAULT_MODEL_PATH):
    """Download test model if not present."""
    config_file = os.path.join(model_path, "config.json")
    model_file = os.path.join(model_path, "model.safetensors")
    
    # Check if model exists and is valid
    if os.path.exists(config_file) and os.path.exists(model_file):
        print(f"   Model already exists at {model_path}")
        return True
    
    print(f"   Downloading test model ({DEFAULT_MODEL_HF})...")
    print("   This may take a few minutes on first run...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Download and save model
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_HF)
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_HF)
        
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        print(f"   Model downloaded and saved to {model_path}")
        return True
        
    except Exception as e:
        print(f"   ERROR downloading model: {e}")
        return False


def test_adapter_import():
    """Test 1: Test that the adapter can be imported correctly."""
    print("\n" + "="*60)
    print("TEST 1: Import Test")
    print("="*60)
    
    try:
        from adapters.offline.openassistant_adapter import OpenAssistantAdapter
        print("   Import successful!")
        
        # Check class inheritance
        from adapters.base import OfflineAdapter
        assert issubclass(OpenAssistantAdapter, OfflineAdapter), \
            "Should be subclass of OfflineAdapter"
        print("   OpenAssistantAdapter is correctly a subclass of OfflineAdapter")
        print("   PASS")
        return True
    except ImportError as e:
        print(f"   FAIL: Import failed - {e}")
        return False


def test_adapter_creation(model_path: str, device: str):
    """Test 2: Test adapter creation."""
    print("\n" + "="*60)
    print("TEST 2: Adapter Creation")
    print("="*60)
    
    from adapters.offline.openassistant_adapter import OpenAssistantAdapter
    from adapters.base import ExecutionMode
    
    adapter = OpenAssistantAdapter(
        name="test-openassistant",
        model_path=model_path,
        device=device,
        model_type="sft",
        estimated_memory_mb=500
    )
    
    print(f"   Adapter created: {adapter}")
    print(f"   Execution mode: {adapter.execution_mode}")
    print(f"   Model type: {adapter.model_type}")
    print(f"   Is loaded: {adapter.is_loaded()}")
    
    assert adapter.execution_mode == ExecutionMode.OFFLINE, "Should be OFFLINE mode"
    assert not adapter.is_loaded(), "Should not be loaded yet"
    print("   PASS")
    return adapter


def test_model_loading(adapter):
    """Test 3: Test model loading."""
    print("\n" + "="*60)
    print("TEST 3: Model Loading")
    print("="*60)
    
    try:
        adapter.load()
        print(f"   Model loaded successfully!")
        print(f"   Is loaded: {adapter.is_loaded()}")
        print(f"   Memory usage: {adapter.get_memory_usage()}")
        
        assert adapter.is_loaded(), "Should be loaded now"
        print("   PASS")
        return True
    except Exception as e:
        print(f"   FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_generation(adapter):
    """Test 4: Test text generation."""
    print("\n" + "="*60)
    print("TEST 4: Text Generation")
    print("="*60)
    
    from adapters.base import GenerationConfig
    
    test_prompt = "What is artificial intelligence?"
    config = GenerationConfig(max_tokens=50, temperature=0.7, do_sample=True)
    
    try:
        response = adapter.generate(test_prompt, config)
        print(f"   Generation successful!")
        print(f"   Prompt: {test_prompt}")
        
        # Truncate response for cleaner output
        resp_text = response.text[:150] + "..." if len(response.text) > 150 else response.text
        print(f"   Response: {resp_text}")
        print(f"   Latency: {response.latency_ms:.1f}ms")
        print(f"   Tokens used: {response.tokens_used}")
        print(f"   Execution mode: {response.execution_mode}")
        print("   PASS")
        return True
    except Exception as e:
        print(f"   FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_offload_to_cpu(adapter):
    """Test 5: Test offloading to CPU."""
    print("\n" + "="*60)
    print("TEST 5: Offload to CPU")
    print("="*60)
    
    try:
        adapter.offload_to_cpu()
        print(f"   Offload successful!")
        print(f"   Device is now: {adapter.device}")
        print("   PASS")
        return True
    except Exception as e:
        print(f"   FAIL: {e}")
        return False


def test_unload(adapter):
    """Test 6: Test model unloading."""
    print("\n" + "="*60)
    print("TEST 6: Model Unload")
    print("="*60)
    
    try:
        adapter.unload()
        print(f"   Unload successful!")
        print(f"   Is loaded: {adapter.is_loaded()}")
        assert not adapter.is_loaded(), "Should not be loaded after unload"
        print("   PASS")
        return True
    except Exception as e:
        print(f"   FAIL: {e}")
        return False


def run_all_tests(model_path: str = DEFAULT_MODEL_PATH, device: str = "cpu"):
    """Run all OpenAssistant adapter tests."""
    
    print("#" * 60)
    print("# OpenAssistant Adapter Test Suite")
    print("#" * 60)
    print(f"\nConfiguration:")
    print(f"  Model path: {model_path}")
    print(f"  Device: {device}")
    
    results = []
    
    # Test 0: Ensure model exists
    print("\n" + "="*60)
    print("SETUP: Checking/Downloading Test Model")
    print("="*60)
    if not ensure_model_exists(model_path):
        print("\nFATAL: Could not setup test model. Aborting tests.")
        return
    
    # Test 1: Import
    results.append(("Import", test_adapter_import()))
    if not results[-1][1]:
        print("\nFATAL: Import failed. Cannot continue tests.")
        return
    
    # Test 2: Creation
    try:
        adapter = test_adapter_creation(model_path, device)
        results.append(("Creation", True))
    except Exception as e:
        print(f"   FAIL: {e}")
        results.append(("Creation", False))
        print("\nFATAL: Creation failed. Cannot continue tests.")
        return
    
    # Test 3: Loading
    load_success = test_model_loading(adapter)
    results.append(("Loading", load_success))
    
    if load_success:
        # Test 4: Generation
        results.append(("Generation", test_text_generation(adapter)))
        
        # Test 5: Offload
        results.append(("Offload", test_offload_to_cpu(adapter)))
        
        # Test 6: Unload
        results.append(("Unload", test_unload(adapter)))
    
    # Summary
    print("\n" + "#" * 60)
    print("# Test Summary")
    print("#" * 60)
    
    passed = 0
    failed = 0
    for name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "[+]" if success else "[-]"
        print(f"  {symbol} {name}: {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
    else:
        print("\nSome tests failed. Check output above for details.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenAssistant Adapter Test Suite")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to test model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use: cpu or cuda (default: cpu)")
    parser.add_argument("--import-only", action="store_true",
                        help="Only test import, skip model loading tests")
    
    args = parser.parse_args()
    
    if args.import_only:
        test_adapter_import()
    else:
        run_all_tests(model_path=args.model_path, device=args.device)
