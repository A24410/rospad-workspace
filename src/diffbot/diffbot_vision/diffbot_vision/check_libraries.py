import importlib.util
import sys

print("Python:", sys.version)

modules = [
    "numpy", "cv2", "torch", "torchvision",
    "onnxruntime", "ultralytics", "PIL"
]

for name in modules:
    try:
        available = importlib.util.find_spec(name) is not None
        print(name, "AVAILABLE" if available else "NOT INSTALLED")
    except Exception as e:
        print(name, "ERROR:", e)