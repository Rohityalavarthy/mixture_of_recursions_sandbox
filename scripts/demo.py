"""
MoR Model Demo Script - Demonstrates both Expert-Choice and Token-Choice routing
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mor import SimpleMoRModel, SyntheticDataGenerator
from mor.train import train_model

def demo_routing_strategy(mor_type="expert_choice"):
 
    print(f"\n{'='*60}")
    print(f"Testing {mor_type.upper().replace('_', '-')} Routing Strategy")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model config
    config = {
        'vocab_size': 1000,
        'hidden_size': 128,
        'num_classes': 4,
        'num_layers': 2,
        'num_recursions': 3,
        'num_heads': 8,
        'max_seq_len': 32,
        'mor_type': mor_type,  # choice bw expert choice or token choice MoR
        'capacity_factor': 0.8,
        'alpha': 0.1,
        'dropout': 0.1
    }

    print(f"\nModel Configuration ({mor_type}):")
    for key, value in config.items():
        print(f"  {key}: {value}")

    model = SimpleMoRModel(**config).to(device)
    print(f"\nModel created with {model.get_num_parameters():,} parameters")

    # Generate data
    data_gen = SyntheticDataGenerator(vocab_size=config['vocab_size'], 
                                     max_seq_len=config['max_seq_len'])

    train_loader = data_gen.create_dataloader(task_type="pattern", num_samples=200, batch_size=16)
    val_loader = data_gen.create_dataloader(task_type="pattern", num_samples=50, batch_size=16)

    # Quick training
    print(f"\nTraining {mor_type} model for 3 epochs...")
    history = train_model(model, train_loader, val_loader, num_epochs=3, learning_rate=1e-3, device=device)

    # Test routing analysis
    print(f"\nRouting Analysis for {mor_type}:")
    model.eval()
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            analysis = model.get_routing_analysis(inputs)

            print(f"  Average routing entropy: {analysis['avg_entropy']:.4f}")

            # Show routing-specific info
            for i, layer_info in enumerate(analysis['layer_routing']):
                if hasattr(layer_info, 'selected_tokens') and layer_info.selected_tokens is not None:
                    print(f"  Layer {i} - Token assignments shape: {layer_info.selected_tokens.shape}")
                if hasattr(layer_info, 'routing_weights') and layer_info.routing_weights is not None:
                    print(f"  Layer {i} - Routing weights shape: {layer_info.routing_weights.shape}")
            break

    return history

def main():
    """Main demo function."""
    print("🚀 Mixture of Recursions (MoR) - Routing Strategy Demo")
    print("="*60)

    # Demo both routing strategies
    expert_history = demo_routing_strategy("expert_choice")
    token_history = demo_routing_strategy("token_choice")

    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Strategy':<15} {'Final Train Acc':<15} {'Final Val Acc':<15} {'Routing Entropy':<15}")
    print("-" * 60)
    print(f"{'Expert-Choice':<15} {expert_history['train_acc'][-1]:<15.2f} {expert_history['val_acc'][-1]:<15.2f} {expert_history['routing_entropy'][-1]:<15.4f}")
    print(f"{'Token-Choice':<15} {token_history['train_acc'][-1]:<15.2f} {token_history['val_acc'][-1]:<15.2f} {token_history['routing_entropy'][-1]:<15.4f}")

    print(f"\n🎉 Demo completed! Key differences:")
    print("• Expert-Choice: Each recursion level selects top-k tokens to process")
    print("• Token-Choice: Each token chooses its own recursion depth")
    print("• Both strategies enable adaptive computation per token")
    print("\nTry different configurations in scripts/visualize.py for deeper analysis!")

if __name__ == "__main__":
    main()
