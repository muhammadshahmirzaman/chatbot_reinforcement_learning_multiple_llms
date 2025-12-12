#!/usr/bin/env python
"""
Quick check if OpenAssistant is ready in the configuration
and if dependencies are met.
"""
import sys
import importlib.util

def check_readiness():
    print("="*60)
    print("OpenAssistant Readiness Check")
    print("="*60)
    
    # 1. Check Config
    print("\n1. Checking Configuration...")
    try:
        sys.path.insert(0, '.')
        from adapters.registry import ModelRegistry
        registry = ModelRegistry.from_config("./config/models_production.yaml")
        
        if "openassistant-falcon-7b-4bit" in registry.list_models():
             print("   ✓ openassistant-falcon-7b-4bit configured")
        else:
             print("   ✗ Model not found in config")
             return False
             
        adapter = registry.get("openassistant-falcon-7b-4bit")
        # Access private attribute just to check config
        if hasattr(adapter, 'load_in_4bit') and adapter.load_in_4bit:
            print("   ✓ 4-bit quantization enabled")
        else:
            print("   ✗ 4-bit quantization NOT enabled in adapter")
            
    except Exception as e:
        print(f"   ✗ Configuration check failed: {e}")
        return False

    # 2. Check Dependencies
    print("\n2. Checking Dependencies...")
    bnb_spec = importlib.util.find_spec("bitsandbytes")
    if bnb_spec:
        print("   ✓ bitsandbytes installed")
    else:
        print("   ✗ bitsandbytes NOT installed (Install via: pip install bitsandbytes)")
        
    acc_spec = importlib.util.find_spec("accelerate")
    if acc_spec:
        print("   ✓ accelerate installed")
    else:
        print("   ✗ accelerate NOT installed")

    # 3. Check Download Status (Partial)
    import os
    print("\n3. Checking Cache...")
    # This is a heuristic check for huggingface cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"   Cache dir: {cache_dir}")
    # We can't easily check specific model without more code, but getting here is good.
    
    print("\n" + "="*60)
    return True

if __name__ == "__main__":
    check_readiness()
