import numpy as np
import torch
from pathlib import Path
from config_hardware import optimize_for_3060

class StreamingDataset:
    """Carga datos por lotes desde disco para ahorrar RAM"""
    def __init__(self, data_dir, chunk_size=1000):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.current_chunk = 0
        
    def load_chunk(self):
        """Carga solo un fragmento del dataset"""
        chunk_file = self.data_dir / f'chunk_{self.current_chunk}.npz'
        data = np.load(chunk_file)
        
        # Convertir a tensores por lotes pequeños
        batch_size = optimize_for_3060()
        for i in range(0, len(data['texts']), batch_size):
            yield {
                'texts': data['texts'][i:i+batch_size],
                'traits': torch.from_numpy(data['traits'][i:i+batch_size]).float()
            }
        
        self.current_chunk = (self.current_chunk + 1) % self.get_total_chunks()
    
    def preprocess_and_chunk(self, raw_data, traits_data):
        """Divide dataset grande en chunks para streaming"""
        n_samples = len(raw_data)
        chunk_size = min(self.chunk_size, n_samples)
        
        for i in range(0, n_samples, chunk_size):
            chunk = {
                'texts': raw_data[i:i+chunk_size],
                'traits': traits_data[i:i+chunk_size]
            }
            np.savez(self.data_dir / f'chunk_{i//chunk_size}.npz', chunk)