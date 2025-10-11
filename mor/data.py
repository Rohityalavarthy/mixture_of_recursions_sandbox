"""
Synthetic Data Generation for MoR Experiments

This module provides utilities for generating synthetic datasets
to test and visualize MoR model behavior.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional

class SyntheticDataGenerator:

    def __init__(self, vocab_size: int = 1000, max_seq_len: int = 64):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def generate_copy_task(self, 
                          num_samples: int = 1000, 
                          copy_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = []
        labels = []

        for _ in range(num_samples):
            seq = torch.randint(1, self.vocab_size, (self.max_seq_len,))
            start_pos = torch.randint(0, self.max_seq_len - copy_length, (1,)).item()
            pattern = torch.randint(1, min(50, self.vocab_size), (copy_length,))
            seq[start_pos:start_pos + copy_length] = pattern
            label = (pattern.sum() % 4).item()

            inputs.append(seq)
            labels.append(label)

        return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)

    def generate_counting_task(self, 
                              num_samples: int = 1000, 
                              target_token: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = []
        labels = []

        for _ in range(num_samples):
            seq = torch.randint(1, self.vocab_size, (self.max_seq_len,))
            count = (seq == target_token).sum().item()
            label = min(count, 3)

            inputs.append(seq)
            labels.append(label)

        return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)

    def generate_pattern_recognition_task(self, 
                                        num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = []
        labels = []

        patterns = [
            lambda: torch.arange(1, 11),
            lambda: torch.arange(10, 0, -1),
            lambda: torch.tensor([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]),
            lambda: torch.tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        ]

        for _ in range(num_samples):
            pattern_idx = torch.randint(0, 4, (1,)).item()
            pattern = patterns[pattern_idx]()
            seq = torch.randint(1, self.vocab_size, (self.max_seq_len,))
            start_pos = torch.randint(0, self.max_seq_len - len(pattern), (1,)).item()
            seq[start_pos:start_pos + len(pattern)] = pattern

            inputs.append(seq)
            labels.append(pattern_idx)

        return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)

    def create_dataloader(self, 
                         task_type: str = "copy", 
                         num_samples: int = 1000, 
                         batch_size: int = 32, 
                         **kwargs) -> DataLoader:
        if task_type == "copy":
            inputs, labels = self.generate_copy_task(num_samples, **kwargs)
        elif task_type == "counting":
            inputs, labels = self.generate_counting_task(num_samples, **kwargs)
        elif task_type == "pattern":
            inputs, labels = self.generate_pattern_recognition_task(num_samples, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        dataset = TensorDataset(inputs, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
