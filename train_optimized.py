import torch

def train_with_memory_constraints(model, train_loader, epochs=30):
    """Entrenamiento optimizado para 12GB VRAM"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Mixed precision solo si CUDA está disponible
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Configuración para 3060
    accumulation_steps = 4  # Simula batch_size efectivo de 16
    gradient_clip = 1.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            # Mover batch al dispositivo
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    # Forward pass con precisión mixta
                    outputs = model(batch['input_ids'], batch['attention_mask'])
                    loss = compute_loss(outputs, batch['traits'])
                    
                    # Escalar pérdida para acumulación
                    scaled_loss = loss / accumulation_steps
                
                # Backward con escalado
                scaler.scale(scaled_loss).backward()
            else:
                # Forward pass normal (CPU)
                outputs = model(batch['input_ids'], batch['attention_mask'])
                loss = compute_loss(outputs, batch['traits'])
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
            
            # Acumular gradientes
            if (i + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Limpieza agresiva solo si CUDA
                if use_amp:
                    torch.cuda.empty_cache()
        
        # Guardar checkpoint ligero
        if epoch % 5 == 0:
            save_checkpoint(model, f"checkpoint_epoch_{epoch}.pt", compressed=True)

def save_checkpoint(model, path, compressed=True):
    """Guarda solo parámetros esenciales"""
    if compressed:
        checkpoint = {
            'state_dict': {k: v.half() for k, v in model.state_dict().items()},  # FP16
            'config': model.config
        }
    torch.save(checkpoint, path)