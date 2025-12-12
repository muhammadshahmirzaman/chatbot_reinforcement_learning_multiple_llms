"""Quick test to verify offline models are working."""
import sys
sys.path.insert(0, '.')

from adapters.registry import ModelRegistry
from adapters.base import GenerationConfig

# Load test config
registry = ModelRegistry.from_config("./config/models_test.yaml")

print(f"✓ Loaded {registry.get_num_models()} models")
print(f"  Models: {registry.list_models()}")

# Test each model
config = GenerationConfig(max_tokens=20)

for model_name in registry.list_models():
    print(f"\nTesting {model_name}...")
    try:
        adapter = registry.get(model_name)
        adapter.load()
        
        response = adapter.generate("What is 2+2?", config)
        print(f"  ✓ Response: {response.text[:50]}...")
        
        adapter.unload()
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n✓ Model verification complete!")
