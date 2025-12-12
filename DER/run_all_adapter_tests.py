#!/usr/bin/env python
"""
Run All Adapter Tests

This script runs all individual adapter test files and provides
a summary of which adapters are working.
"""

import subprocess
import sys

def run_test(test_file, model_name):
    """Run a single test file and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=False,
            timeout=120  # 2 minute timeout per test
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"✗ {model_name} test timed out!")
        return False
    except Exception as e:
        print(f"✗ {model_name} test failed: {e}")
        return False

def main():
    """Run all adapter tests."""
    print("="*60)
    print("DER ADAPTER TEST SUITE")
    print("="*60)
    
    # Define all tests
    tests = [
        ("test_opt125m_adapter.py", "OPT-125M"),
        ("test_tinyllama_adapter.py", "TinyLlama 1.1B"),
        ("test_phi2_adapter.py", "Phi-2"),
        ("test_phi3mini_adapter.py", "Phi-3 Mini"),
        ("test_cached_adapter.py", "Cached Baseline"),
    ]
    
    results = {}
    
    # Run each test
    for test_file, model_name in tests:
        success = run_test(test_file, model_name)
        results[model_name] = success
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        symbol = "✅" if success else "❌"
        print(f"{symbol} {model_name:20s} {status}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 ALL ADAPTERS WORKING!")
        return 0
    else:
        print(f"\n⚠️  {failed} adapter(s) need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
