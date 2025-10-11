"""
Mixture of Recursions (MoR) - A Lightweight Implementation

This package provides a clean, modular implementation of the Mixture of Recursions
architecture for research and experimentation.
"""

from .models import SimpleMoRModel, MoRConfig, ExpertChoiceMoRLayer, TokenChoiceMoRLayer
from .data import SyntheticDataGenerator
from .train import train_model, evaluate_model

__version__ = "1.0.0"
__author__ = "Rohit Yalavarthy"

__all__ = [
    "SimpleMoRModel",
    "MoRConfig", 
    "ExpertChoiceMoRLayer",
    "TokenChoiceMoRLayer",
    "SyntheticDataGenerator",
    "train_model",
    "evaluate_model"
]
