import torch
import gc

def optimize_for_3060():
    """Configuración agresiva para hardware limitado"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # Configuración de precisión mixta (FP16)
        torch.set_float32_matmul_precision('medium')
        
        # Limitar uso de RAM del sistema
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Batch size dinámico basado en memoria disponible
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        batch_size = max(1, int(free_memory / (1024**3) * 4))  # ~4 samples por GB libre
    else:
        print("ADVERTENCIA: CUDA no disponible. Usando CPU.")
        gc.collect()
        batch_size = 4  # Batch size conservativo para CPU
    
    return batch_size