import subprocess
import sys

def run_command(cmd):
    print(f"Ejecutando: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"OK: {result.stdout[:100]}...")
    return result.returncode == 0

print("=== REPARANDO IMPORTS ===\n")

# 1. Verificar pip
run_command("python -m pip install --upgrade pip")

# 2. Instalar transformers específicamente
run_command("pip install transformers==4.41.2")

# 3. Instalar tokenizers
run_command("pip install tokenizers==0.19.1")

# 4. Verificar
print("\n=== VERIFICACIÓN FINAL ===")
run_command("python -c \"from transformers import DistilBertTokenizer; print('Tokenizer OK')\"")
run_command("python -c \"import torch; print(f'PyTorch {torch.__version__}')\"")