# src/training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Ensure root directory is in path so we can import modules cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_pipeline.dataset import CelebDFDataset, baseline_transforms
from src.modules.model import ResilientDetector
from src.utils.metrics import calculate_metrics

def train():
    # 1. Hardware Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training initialized on: {device}")

    # 2. Configure Paths (Assuming execution from the root ResilientDeep/ folder)
    data_dir = os.path.abspath("data/sample_dataset") 
    checkpoint_dir = os.path.abspath("models/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True) # Create checkpoint folder if missing
    
    print("Indexing dataset... This ensures O(1) batch loading.")
    dataset = CelebDFDataset(
        root_dir=data_dir, 
        real_folder="Celeb-real",       
        fake_folder="Celeb-synthesis",  
        transform=baseline_transforms
    )
    
    if len(dataset) == 0:
        print(f"Error: Dataset is empty. Check your folder path: {data_dir}")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 3. Model & Optimization Setup
    model = ResilientDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The O(N) Training Loop with Evaluation
    epochs = 3
    best_f1 = 0.0 # Track the best score to save the optimal weights

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Lists to keep track of predictions for metrics
        all_preds = []
        all_labels = []
        
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Record for F1/Accuracy calculation
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu())
            all_labels.extend(labels.cpu())
            
            if i % 10 == 0: 
                print(f"Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}")

        # --- Epoch Evaluation & Saving ---
        acc, f1 = calculate_metrics(torch.tensor(all_preds), torch.tensor(all_labels))
        print(f"Epoch {epoch+1} Results -> Loss: {running_loss/len(dataloader):.4f} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        
        # Save standard checkpoint
        save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_save_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_save_path)
            print(f"*** New Best Model Saved! (F1: {best_f1:.4f}) ***")

    print("Training Complete. Prototype Baseline Established.")

if __name__ == "__main__":
    train()