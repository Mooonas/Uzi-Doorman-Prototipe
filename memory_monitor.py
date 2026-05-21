import psutil
import pynvml

class HardwareMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.cuda_available = True
        except:
            print("ADVERTENCIA: No se pudo inicializar NVML. Monitoreo de GPU deshabilitado.")
            self.cuda_available = False
        
    def get_memory_status(self):
        # Memoria RAM
        ram = psutil.virtual_memory()
        ram_used = ram.used / 1024**3
        ram_total = ram.total / 1024**3
        
        result = {
            'ram_used_gb': round(ram_used, 2),
            'ram_total_gb': round(ram_total, 2),
            'ram_percent': round(ram_used/ram_total*100, 1)
        }
        
        # Memoria GPU si está disponible
        if self.cuda_available:
            try:
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                gpu_used = gpu_info.used / 1024**3
                gpu_total = gpu_info.total / 1024**3
                
                result.update({
                    'gpu_used_gb': round(gpu_used, 2),
                    'gpu_total_gb': round(gpu_total, 2),
                    'gpu_percent': round(gpu_used/gpu_total*100, 1)
                })
            except:
                result.update({
                    'gpu_used_gb': 0,
                    'gpu_total_gb': 0,
                    'gpu_percent': 0
                })
        else:
            result.update({
                'gpu_used_gb': 0,
                'gpu_total_gb': 0,
                'gpu_percent': 0
            })
        
        return result
    
    def adaptive_batch_size(self, current_batch_size):
        """Ajusta batch size dinámicamente"""
        if not self.cuda_available:
            return current_batch_size
            
        status = self.get_memory_status()
        
        if status['gpu_percent'] > 90:
            return max(1, current_batch_size // 2)
        elif status['gpu_percent'] < 70:
            return min(32, current_batch_size * 2)
        return current_batch_size