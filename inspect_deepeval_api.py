
import inspect

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

print("=" * 65)
print("DEEPEVAL CUSTOM LLM METHOD INSPECTION")
print("=" * 65)

methods = [
    "load_model",
    "generate",
    "a_generate",
    "get_model_name"
]

for method_name in methods:
    method = getattr(DeepEvalBaseLLM, method_name)

    print(f"\n{method_name}:")
    print(inspect.signature(method))

print("\n" + "=" * 65)
print("LLM TEST CASE FIELDS")
print("=" * 65)

print(inspect.signature(LLMTestCase))

print("\nInspection completed successfully!")
