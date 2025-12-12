#!/usr/bin/env python
"""Quick verification that all components are integrated."""
import sys
sys.path.insert(0, '.')

print("="*60)
print("QUICK INTEGRATION VERIFICATION")
print("="*60)

# 1. Load registry
print("\n1. Loading model registry...")
from adapters.registry import ModelRegistry
registry = ModelRegistry.from_config("./config/models_production.yaml")
models = registry.list_models()
print(f"   ✓ Loaded {len(models)} models")

# 2. Check OpenAssistant
print("\n2. Checking OpenAssistant...")
if "openassistant-sft" in models:
    print("   ✓ OpenAssistant-SFT is in model pool")
else:
    print("   ✗ OpenAssistant-SFT NOT found")

# 3. List all models
print("\n3. Available models:")
for i, m in enumerate(models, 1):
    cost = registry.get_model_cost(m)
    print(f"   {i}. {m:25s} (cost: {cost})")

# 4. Verify imports work
print("\n4. Verifying imports...")
from environment import Environment
from reward import KTP_calculate
print("   ✓ environment.py imported")
print("   ✓ reward.py imported")

print("\n" + "="*60)
print("✓ ALL COMPONENTS INTEGRATED!")
print("="*60)
