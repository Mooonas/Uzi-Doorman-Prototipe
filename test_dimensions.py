import torch
from transformers import DistilBertTokenizer
from uzi_core_light import PersonalityEmbeddingLite
from decision_engine_lite import DecisionEngineLite

print("=== TEST DE DIMENSIONES ===")

# 1. Crear modelo
model = PersonalityEmbeddingLite(hidden_size=384)
print(f"1. PersonalityEmbeddingLite creado")
print(f"   Trait matrix: {model.trait_matrix.shape}")  # [5, 384]
print(f"   Context fusion input: {model.context_fusion[0].in_features}")
print(f"   Context fusion output: {model.context_fusion[0].out_features}")

# 2. Test forward pass
print("\n2. Cargando tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

test_input = "Los sistemas humanos son predecibles"
inputs = tokenizer(test_input, return_tensors='pt', truncation=True, max_length=256, padding=True)

print(f"   Tokenizado: '{test_input[:30]}...'")
print(f"   Input IDs shape: {inputs['input_ids'].shape}")
print(f"   Attention mask shape: {inputs['attention_mask'].shape}")

with torch.no_grad():
    embedding = model(inputs['input_ids'], inputs['attention_mask'])
    print(f"   Output embedding shape: {embedding.shape}")  # [1, 384]

# 3. Decision engine
print("\n3. Probando motor de decisiones...")
decision_engine = DecisionEngineLite(trait_dims=5, embedding_dim=384)
traits = torch.tensor([[0.5, 0.6, 0.7, 0.8, 0.9]])

# Asegurar que embedding tenga dimensión de secuencia [batch, seq_len, features]
embedding_with_seq = embedding.unsqueeze(1)  # [1, 1, 384]
decision = decision_engine(traits, embedding_with_seq)

print(f"   Traits shape: {traits.shape}")
print(f"   Embedding with seq shape: {embedding_with_seq.shape}")
print(f"   Decision shape: {decision.shape}")  # [1, 4]

# 4. Interpretación
decision_types = ["CONFRONTAR", "IGNORAR", "ANALIZAR", "SABOTEAR"]
decision_idx = decision.argmax(dim=1).item()

print(f"\n4. Interpretación:")
print(f"   Decisión: {decision_types[decision_idx]}")
print(f"   Probabilidades: {decision.cpu().numpy()[0].round(3)}")

print("\n✅ Todas las dimensiones son compatibles")