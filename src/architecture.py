# Contenido para el nuevo archivo: src/architecture.py

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader as TorchDataLoader
from PIL import Image
from tqdm.notebook import tqdm

# --- Clase Dataset ---
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, label_map, trans=None):
        self.X = X; self.y = y; self.trans = trans; self.label_map = label_map
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, ix):
        try:
            img = Image.open(self.X[ix]).convert('RGB')
            if self.trans:
                img = self.trans(img)
            # La transformación a tensor se hace aquí para asegurar que se aplica siempre
            img_tensor = transforms.ToTensor()(img)
            label = torch.tensor(self.label_map[self.y[ix]])
            return img_tensor, label
        except Exception:
            return None, None

# --- Función Collate ---
def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

# --- Arquitectura del Modelo ---
class Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    def freeze(self):
        print("Congelando capas convolucionales (Transfer Learning)...")
        for param in self.resnet.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        print("Descongelando todas las capas (Fine-tuning)...")
        for param in self.resnet.parameters():
            param.requires_grad = True

# --- Función de Entrenamiento (con Early Stopping) ---
def fit(model, dataloader, epochs, lr, patience, model_name="model", device="cuda"):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = f'{model_name}_best.pth'

    print(f"\n--- Entrenando {model_name} por un máximo de {epochs} epochs (Paciencia: {patience}) ---")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_acc = [], []
        bar = tqdm(dataloader['train'], desc=f"Epoch {epoch}/{epochs} [Train]")
        for X, y in bar:
            if X.nelement() == 0: continue
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            
            acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
            train_loss.append(loss.item())
            train_acc.append(acc)
            bar.set_postfix(loss=f"{np.mean(train_loss):.4f}", acc=f"{np.mean(train_acc):.4f}")
        history['train_loss'].append(np.mean(train_loss))

        model.eval()
        val_loss, val_acc = [], []
        bar = tqdm(dataloader['valid'], desc=f"Epoch {epoch}/{epochs} [Valid]")
        with torch.no_grad():
            for X, y in bar:
                if X.nelement() == 0: continue
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
                val_loss.append(loss.item())
                val_acc.append(acc)
                bar.set_postfix(val_loss=f"{np.mean(val_loss):.4f}", val_acc=f"{np.mean(val_acc):.4f}")
        
        current_val_loss = np.mean(val_loss)
        history['val_loss'].append(current_val_loss)
        history['val_acc'].append(np.mean(val_acc))
        
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch}: Mejora en Val Loss. Guardando mejor modelo en '{best_model_path}'")
        else:
            patience_counter += 1
            print(f"Epoch {epoch}: No hubo mejora. Contador de paciencia: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\n¡Early Stopping! No hubo mejora en {patience} épocas.")
            break
            
    print(f"Entrenamiento finalizado. Cargando mejor modelo desde '{best_model_path}'")
    model.load_state_dict(torch.load(best_model_path))
    return history
