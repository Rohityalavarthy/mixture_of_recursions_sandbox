"""
Learn MoR the Easy Way | Lightweight PyTorch implementation of Mixture-of-Recursions with Expert-Choice & Token-Choice routing | Perfect for students, researchers & rapid prototyping | Runs on your laptop!
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
