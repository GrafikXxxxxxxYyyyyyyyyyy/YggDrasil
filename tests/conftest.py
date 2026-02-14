"""Shared test configuration — mocks out TensorFlow to avoid Metal conflict on macOS."""
import sys
import types
import os

# Must happen before any other imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

for mod_name in [
    "tensorflow", "tensorflow.python", "tensorflow.python.framework",
    "tensorflow.python.framework.load_library",
]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        m.__spec__ = None
        m.__path__ = []
        sys.modules[mod_name] = m
