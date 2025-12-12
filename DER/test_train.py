#!/usr/bin/env python
"""
Test Training Script - DER with Offline Models

This tests the DER training pipeline using offline/cached models
that don't require an API server.
"""

import argparse
import sys
import json
import os

# Pre-populate cache with test responses
def setup_test_cache():
    """Create cached responses for testing."""
    os.makedirs("./cache", exist_ok=True)
    
    # Cache 1: Simple direct answers
    cache1 = {
        # Hash format: just use simple keys for testing
        "q1": "The answer is 42.",
        "q2": "Paris is the capital of France.",
        "q3": "Machine learning is a subset of AI."
    }
    
    # Cache 2: Detailed answers  
    cache2 = {
        "q1": "The answer to life, universe, and everything is 42, according to Douglas Adams.",
        "q2": "Paris, located in northern France, is the capital and most populous city.",
        "q3": "Machine learning is a field of AI that uses statistical techniques to give computer systems the ability to learn from data."
    }
    
    # Cache 3: Brief answers
    cache3 = {
        "q1": "42",
        "q2": "Paris",
        "q3": "ML is AI that learns from data"
    }
    
    with open("./cache/test_cache_1.json", "w") as f:
        json.dump(cache1, f, indent=2)
    
    with open("./cache/test_cache_2.json", "w") as f:
        json.dump(cache2, f, indent=2)
        
    with open("./cache/test_cache_3.json", "w") as f:
        json.dump(cache3, f, indent=2)
    
    print("✓ Test caches created")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("DER Test Training")
    parser.add_argument("--setup-only", action="store_true", help="Only setup test environment")
    args = parser.parse_args()
    
    # Setup test environment
    print("Setting up test environment...")
    setup_test_cache()
    
    if args.setup_only:
        print("\nTest environment ready!")
        print("\nTo run training with test config:")
        print("  python train_ppo_multi.py \\")
        print("    --model_config ./config/models_test.yaml \\")
        print("    --epochs 2 \\")
        print("    --batch_size 2 \\")
        print("    --max_train_data_size 10 \\")
        print("    --device cpu")
        sys.exit(0)
    
    # Run training
    print("\nStarting training with test models...")
    import subprocess
    
    cmd = [
        sys.executable, "train_ppo_multi.py",
        "--model_config", "./config/models_test.yaml",
        "--epochs", "2",
        "--batch_size", "2",
        "--max_train_data_size", "10",
        "--device", "cpu",
        "--thread_nums", "2"
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)
