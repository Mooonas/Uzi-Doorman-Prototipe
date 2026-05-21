import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uzi_core_light import PersonalityEmbeddingLite
from decision_engine_lite import DecisionEngineLite
from memory_monitor import HardwareMonitor

def main():
    print("=== UZI DOORMAN - Sistema de Personalidad y Decisiones ===\n")
    
    # 1. Verificar CUDA
    device = check_cuda()
    
    # 2. Cargar configuración
    try:
        from config import CONFIG as config
        print("Configuración cargada")
    except ImportError:
        print("Creando configuración por defecto...")
        config = {
            'model_architecture': {'hidden_size': 384},
            'training': {'batch_size': 4}
        }
    
    # 3. Inicializar monitor
    monitor = HardwareMonitor()
    status = monitor.get_memory_status()
    print(f"\nEstado hardware:")
    print(f"GPU: {status['gpu_used_gb']}/{status['gpu_total_gb']} GB ({status['gpu_percent']}%)")
    print(f"RAM: {status['ram_used_gb']}/{status['ram_total_gb']} GB ({status['ram_percent']}%)")
    
    # 4. Inicializar modelo
    print("\nInicializando modelo...")
    try:
        model = PersonalityEmbeddingLite(
            trait_dims=5,
            hidden_size=config['model_architecture']['hidden_size']
        ).to(device)
        print("✓ Modelo cargado correctamente")
        
        # Obtener dimensiones reales
        model_config = model.get_config()
        print(f"  Dimensiones: BERT={model_config['bert_dim']}, "
              f"Traits={model_config['trait_dims']}, "
              f"Hidden={model_config['hidden_size']}")
              
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Sistema de decisiones CON DIMENSIONES CORRECTAS
    print("\nInicializando motor de decisiones...")
    try:
        # Usar embedding_dim del modelo
        embedding_dim = model_config['hidden_size']
        decision_engine = DecisionEngineLite(
            trait_dims=5, 
            embedding_dim=embedding_dim
        ).to(device)
        print(f"✓ Motor inicializado (embedding_dim={embedding_dim})")
    except Exception as e:
        print(f"✗ Error en motor: {e}")
        return
    
    # 6. Ejemplo de inferencia VERIFICADO
    print("\nProbando inferencia...")
    test_text = "Los sistemas humanos son predecibles"
    
    # Tokenizar
    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    inputs = tokenizer(
        test_text, 
        return_tensors='pt', 
        truncation=True, 
        max_length=256,
        padding=True
    )
    
    # Mover a dispositivo
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inferencia con verificación
    with torch.no_grad():
        print(f"\nFormas de entrada:")
        print(f"  input_ids: {inputs['input_ids'].shape}")
        print(f"  attention_mask: {inputs['attention_mask'].shape}")
        
        # Obtener embedding
        embedding = model(inputs['input_ids'], inputs['attention_mask'])
        print(f"  Embedding output: {embedding.shape}")
        
        # Crear rasgos de prueba
        traits = torch.tensor([[0.8, 0.7, 0.9, 0.6, 0.8]], device=device)
        print(f"  Traits: {traits.shape}")
        
        # Decisión
        decision = decision_engine(traits, embedding.unsqueeze(1))  # Añadir dimensión seq_len
        print(f"  Decision output: {decision.shape}")
    
    # 7. Interpretar resultados
    decision_types = ["CONFRONTAR", "IGNORAR", "ANALIZAR", "SABOTEAR"]
    decision_idx = decision.argmax(dim=1).item()
    confidence = decision[0, decision_idx].item()
    
    print(f"\n=== RESULTADO DE INFERENCIA ===")
    print(f"Texto: '{test_text}'")
    print(f"Decisión: {decision_types[decision_idx]} (confianza: {confidence:.2%})")
    print(f"Distribución completa: {decision.cpu().numpy()[0].round(3)}")
    
    # 8. Test adicional
    print(f"\n=== TEST DE MEMORIA ===")
    model.update_memory(embedding[0])
    print(f"Memoria actualizada (posición {model.memory_ptr})")
    
    print("\n✓ Sistema operativo completamente")

def check_cuda():
    """Verificación robusta de CUDA"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        
        print(f"✓ CUDA disponible: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    else:
        print("⚠ Ejecutando en CPU")
        return torch.device('cpu')

if __name__ == "__main__":
    main()