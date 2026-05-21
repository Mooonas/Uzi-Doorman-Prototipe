import sys
print(f"Python: {sys.version}\n")

# Verificar todos los imports
imports_to_check = [
    'torch',
    'torch.nn',
    'transformers',
    'psutil',
    'nvidia.ml'  # nvidia-ml-py
]

for imp in imports_to_check:
    try:
        if '.' in imp:
            # Para imports como 'nvidia.ml'
            parts = imp.split('.')
            module = __import__(parts[0])
            for part in parts[1:]:
                module = getattr(module, part)
            print(f"✓ {imp}")
        else:
            __import__(imp)
            print(f"✓ {imp}")
    except ImportError as e:
        print(f"✗ {imp}: {e}")

# Verificar CUDA específicamente
print("\n=== CUDA STATUS ===")
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA versión: {torch.version.cuda}")
    print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")