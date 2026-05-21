import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.checkpoint import checkpoint

class PersonalityEmbeddingLite(nn.Module):
    def __init__(self, trait_dims=5, hidden_size=384):
        super().__init__()
        
        # Modelo base - DistilBERT tiene 768 dimensiones de salida
        self.language_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Rasgos psicológicos - dimensiones deben coincidir
        self.trait_matrix = nn.Parameter(torch.randn(trait_dims, hidden_size))
        
        # CORRECCIÓN CRÍTICA: DistilBERT output = 768, traits = hidden_size
        # Entrada: 768 (BERT) + hidden_size (traits) = 768 + 384 = 1152
        self.context_fusion = nn.Sequential(
            nn.Linear(768 + hidden_size, hidden_size),  # CAMBIADO: 1152→384
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Memoria
        self.register_buffer('memory_buffer', torch.zeros(50, hidden_size))
        self.memory_ptr = 0
        
    def forward(self, input_ids, attention_mask):
        # Forward con checkpointing
        def _forward(input_ids, attention_mask):
            outputs = self.language_model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            return outputs.last_hidden_state[:,0,:]  # Shape: [batch, 768]
        
        base_emb = checkpoint(_forward, input_ids, attention_mask, use_reentrant=False)
        
        # Verificar dimensiones
        batch_size = base_emb.size(0)
        
        # Vector de personalidad: [hidden_size] -> expandir a [batch, hidden_size]
        personality_vec = self.trait_matrix.mean(dim=0)  # Shape: [hidden_size]
        personality_vec_expanded = personality_vec.unsqueeze(0).expand(batch_size, -1)
        
        # Concatenar: [batch, 768] + [batch, 384] = [batch, 1152]
        fused = torch.cat([base_emb, personality_vec_expanded], dim=1)
        
        return self.context_fusion(fused)
    
    def update_memory(self, embedding):
        self.memory_buffer[self.memory_ptr] = embedding.detach().cpu()
        self.memory_ptr = (self.memory_ptr + 1) % self.memory_buffer.size(0)

    def get_config(self):
        return {
            'trait_dims': self.trait_matrix.size(0),
            'hidden_size': self.trait_matrix.size(1),
            'bert_dim': 768
        }