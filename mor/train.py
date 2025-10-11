"""
Training and Evaluation Utilities for MoR Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

def train_model(model, train_loader, val_loader=None, num_epochs=10, learning_rate=1e-3, device="cpu"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'routing_entropy': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        routing_entropies = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, routing_info = model(inputs, return_routing_info=True)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Calculate routing entropy
            if routing_info:
                total_entropy = 0
                for info in routing_info:
                    if hasattr(info, 'router_logits') and info.router_logits is not None:
                        probs = torch.softmax(info.router_logits, dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                        total_entropy += entropy
                avg_entropy = total_entropy / len(routing_info) if routing_info else 0
                routing_entropies.append(avg_entropy.item())

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / train_total
        epoch_routing_entropy = np.mean(routing_entropies) if routing_entropies else 0

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['routing_entropy'].append(epoch_routing_entropy)

        if val_loader:
            val_loss, val_acc = evaluate_model(model, val_loader, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Routing Entropy: {epoch_routing_entropy:.4f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, Routing Entropy: {epoch_routing_entropy:.4f}')

    return history

def evaluate_model(model, data_loader, device="cpu"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

def analyze_routing_patterns(model, data_loader, device="cpu", num_samples=100):
    model.eval()
    model = model.to(device)

    all_routing_weights = []
    all_entropies = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            if i * data_loader.batch_size >= num_samples:
                break

            inputs = inputs.to(device)
            analysis = model.get_routing_analysis(inputs)

            for layer_info in analysis['layer_routing']:
                if hasattr(layer_info, 'routing_weights') and layer_info.routing_weights is not None:
                    weights = layer_info.routing_weights
                    all_routing_weights.append(weights.cpu())
                if hasattr(layer_info, 'router_logits') and layer_info.router_logits is not None:
                    probs = torch.softmax(layer_info.router_logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                    all_entropies.append(entropy.cpu())

    if all_routing_weights:
        all_weights = torch.cat(all_routing_weights, dim=0)
        analysis_results = {
            'mean_routing_weights': all_weights.mean(dim=(0, 1)),
            'std_routing_weights': all_weights.std(dim=(0, 1)),
            'routing_distribution': all_weights.reshape(-1, all_weights.size(-1))
        }
    else:
        analysis_results = {'mean_routing_weights': None, 'std_routing_weights': None, 'routing_distribution': None}

    if all_entropies:
        all_entropies = torch.stack(all_entropies)
        analysis_results['mean_entropy'] = all_entropies.mean().item()
        analysis_results['std_entropy'] = all_entropies.std().item()
    else:
        analysis_results['mean_entropy'] = 0
        analysis_results['std_entropy'] = 0

    return analysis_results
