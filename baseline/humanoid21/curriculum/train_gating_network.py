#!/usr/bin/env python3
"""Script to train a Gating MLP classifier.

Loads collected gating data (.npz), performs a train/validation split, 
trains a binary classification MLP using PyTorch, and exports the model 
weights alongside evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Define Gating MLP network architecture
class GatingMLP(nn.Module):
    """Multi-layer Perceptron to predict safety state probability from robot observations."""

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))  # Add LayerNorm for training stability
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))      # Small dropout to prevent overfitting
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting raw logits."""
        return self.network(x)

    @torch.no_grad()
    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probability in [0, 1] (1 = safe, 0 = unsafe/falling)."""
        return torch.sigmoid(self.network(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gating MLP classifier.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_data",
        help="Directory containing gating_data.npz and summary.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_model",
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden layers dimensions"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float, float, float]:
    """Evaluate loss, accuracy, precision, and recall on the unsafe (0) class."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            correct += (preds == y).sum().item()
            total += len(y)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    avg_loss = total_loss / total
    accuracy = correct / total
    
    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)
    
    # Calculate recall/precision on the Unsafe (0.0) class
    # Unsafe prediction is preds == 0, actual unsafe is targets == 0
    true_unsafe = (targets_np == 0.0)
    pred_unsafe = (preds_np == 0.0)
    
    tp_unsafe = np.sum(true_unsafe & pred_unsafe)
    fp_unsafe = np.sum((~true_unsafe) & pred_unsafe)
    fn_unsafe = np.sum(true_unsafe & (~pred_unsafe))
    
    precision_unsafe = tp_unsafe / (tp_unsafe + fp_unsafe + 1e-8)
    recall_unsafe = tp_unsafe / (tp_unsafe + fn_unsafe + 1e-8)
    
    return avg_loss, accuracy, precision_unsafe, recall_unsafe


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    npz_path = data_dir / "gating_data.npz"
    if not npz_path.exists():
        print(f"Error: dataset file not found at {npz_path}")
        print("Please run the collection script first to generate data.")
        sys.exit(1)
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load dataset
    print("=" * 70)
    print(f"📦 Loading Dataset from {npz_path}")
    data = np.load(npz_path)
    X_raw = data["observations"]
    Y_raw = data["labels"]
    
    num_samples = len(X_raw)
    obs_dim = X_raw.shape[1]
    
    num_pos = np.sum(Y_raw == 1.0)
    num_neg = np.sum(Y_raw == 0.0)
    pos_ratio = num_pos / num_samples * 100.0
    neg_ratio = num_neg / num_samples * 100.0
    
    print(f"   - Total frames:     {num_samples:,}")
    print(f"   - Observation Dim:  {obs_dim}")
    print(f"   - Safe Frames (1):  {num_pos:,} ({pos_ratio:.1f}%)")
    print(f"   - Unsafe Frames (0): {num_neg:,} ({neg_ratio:.1f}%)")
    print(f"   - Hardware Device:  {device}")
    print("=" * 70, flush=True)
    
    # 2. Train-Val Split (shuffle indexes)
    shuffled_idx = np.random.permutation(num_samples)
    val_size = int(num_samples * args.val_ratio)
    train_idx = shuffled_idx[val_size:]
    val_idx = shuffled_idx[:val_size]
    
    X_train = torch.as_tensor(X_raw[train_idx], dtype=torch.float32)
    Y_train = torch.as_tensor(Y_raw[train_idx], dtype=torch.float32)
    X_val = torch.as_tensor(X_raw[val_idx], dtype=torch.float32)
    Y_val = torch.as_tensor(Y_raw[val_idx], dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Define Model, Optimizer, and Loss
    model = GatingMLP(input_dim=obs_dim, hidden_dims=args.hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Calculate pos_weight for loss balance:
    # Since class 1 (Safe) has more samples, we set pos_weight = num_neg / num_pos.
    # This reduces the loss scale for positive samples to match the negative ones.
    pos_weight_val = num_neg / num_pos
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print(f"🚀 Initializing Gating MLP architecture: {obs_dim} -> {args.hidden_dims} -> 1")
    print(f"   - Class Balancing Weight (pos_weight): {pos_weight_val:.4f}")
    print(f"   - Dropout: 10% | Weight Decay: 1e-5")
    print(f"   - Training parameters: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    print("-" * 70, flush=True)
    
    # 4. Training Loop
    best_val_loss = float("inf")
    best_metrics = {}
    
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch).squeeze(1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)
            
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Evaluate on validation set
        val_loss, val_acc, val_precision_unsafe, val_recall_unsafe = evaluate_model(
            model, val_loader, criterion, device
        )
        
        # Log progress every epoch
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"Epoch [{epoch:3d}/{args.epochs:3d}] | "
                f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc*100.1:.1f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100.1:.1f}% | "
                f"Unsafe-Precision: {val_precision_unsafe*100.1:.1f}% Unsafe-Recall: {val_recall_unsafe*100.1:.1f}%",
                flush=True
            )
            
        # Save the best model checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "unsafe_precision": val_precision_unsafe,
                "unsafe_recall": val_recall_unsafe,
                "train_loss": avg_train_loss,
                "train_acc": train_acc
            }
            # Save state dict
            torch.save({
                "epoch": epoch,
                "obs_dim": obs_dim,
                "hidden_dims": args.hidden_dims,
                "state_dict": model.state_dict(),
                "metrics": best_metrics,
            }, output_dir / "gating_model.pt")
            
    # 5. Export and wrap up
    t_train = time.perf_counter() - t_start
    print("=" * 70)
    print("🎉 Gating MLP Model Training Completed Successfully!")
    print(f"   - Saved Model Path: {output_dir / 'gating_model.pt'}")
    print(f"   - Best epoch:      {best_metrics['epoch']} (lowest validation loss)")
    print(f"   - Train Accuracy:  {best_metrics['train_acc']*100.0:.2f}% (Loss: {best_metrics['train_loss']:.4f})")
    print(f"   - Val Accuracy:    {best_metrics['val_acc']*100.0:.2f}% (Loss: {best_metrics['val_loss']:.4f})")
    print(f"   - Critical Gating Metrics (Unsafe/Falling Class):")
    print(f"     - Precision:     {best_metrics['unsafe_precision']*100.0:.2f}% (95%+ indicates minimal false-positives)")
    print(f"     - Recall:        {best_metrics['unsafe_recall']*100.0:.2f}% (98%+ is highly ideal for safety guarantee)")
    print(f"   - Total Training Time: {t_train:.1f} seconds")
    
    # Save a JSON file with model metadata and best metrics
    gating_config = {
        "input_dim": obs_dim,
        "hidden_dims": args.hidden_dims,
        "best_metrics": best_metrics,
        "device": str(device),
        "export_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(output_dir / "gating_config.json", "w", encoding="utf-8") as f:
        json.dump(gating_config, f, indent=4, ensure_ascii=False)
    print(f"   - Saved Metadata:   {output_dir / 'gating_config.json'}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
