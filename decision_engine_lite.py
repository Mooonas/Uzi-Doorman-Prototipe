import torch
import torch.nn as nn

class DecisionEngineLite(nn.Module):
    def __init__(self, trait_dims=5, embedding_dim=384):  # Añadir embedding_dim
        super().__init__()
        
        # Umbrales fijos
        self.register_buffer('trait_thresholds', torch.tensor([0.7, 0.6, 0.8, 0.5, 0.9]))
        
        # Proyector para reducir dimensionalidad si es necesario
        # embedding_dim viene de PersonalityEmbeddingLite (384)
        self.projector = nn.Linear(embedding_dim, 128)  # 384 → 128
        
        # Capa de decisión: traits (5) + projected (128) = 133
        self.decision_layers = nn.Sequential(
            nn.Linear(trait_dims + 128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, 4)  # 4 tipos de decisión
        )
    
    def forward(self, current_traits, situation_embedding):
        # current_traits: [batch, 5]
        # situation_embedding: [batch, seq_len, embedding_dim]
        
        # Proyectar embedding situacional
        situation_mean = situation_embedding.mean(dim=1)  # [batch, embedding_dim]
        projected_situation = self.projector(situation_mean)  # [batch, 128]
        
        # Aplicar umbrales a rasgos
        trait_impulses = current_traits * self.trait_thresholds
        
        # Combinar
        combined = torch.cat([trait_impulses, projected_situation], dim=1)
        
        # Decisión
        logits = self.decision_layers(combined)
        return torch.softmax(logits, dim=1)