"""
Mixture of Recursions (MoR) Models

This module contains the core implementation of the MoR architecture,
including both expert-choice and token-choice routing strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Union
from dataclasses import dataclass

# random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class MoRConfig:
    vocab_size: int = 1000
    hidden_size: int = 128
    num_classes: int = 4
    num_layers: int = 4
    num_recursions: int = 3
    num_heads: int = 8
    max_seq_len: int = 64
    mor_type: str = "expert_choice"  # "expert_choice" or "token_choice"
    capacity_factor: float = 0.8
    alpha: float = 0.1
    dropout: float = 0.1

@dataclass
class MoROutput:
    hidden_states: torch.Tensor
    router_logits: Optional[torch.Tensor] = None
    selected_tokens: Optional[torch.Tensor] = None
    routing_weights: Optional[torch.Tensor] = None
    load_balancing_loss: Optional[torch.Tensor] = None

class RouterBase(nn.Module):

    def __init__(self, hidden_size: int, num_experts: Optional[int] = None, initializer_range: float = 0.02):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts or 1
        self.initializer_range = initializer_range

    def forward(self, x):
        raise NotImplementedError

class LinearRouter(RouterBase):

    def __init__(self, hidden_size: int, num_experts: int = 1, initializer_range: float = 0.02):
        super().__init__(hidden_size, num_experts, initializer_range)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.router.weight.data.normal_(mean=0.0, std=initializer_range)

    def forward(self, x):
        return self.router(x)

class MLPRouter(RouterBase):

    def __init__(self, hidden_size: int, num_experts: int = 1, initializer_range: float = 0.02):
        super().__init__(hidden_size, num_experts, initializer_range)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_experts, bias=False)
        )
        # Initialize weights
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=initializer_range)

    def forward(self, x):
        return self.router(x)

class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, intermediate_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size or (hidden_size * 4)

        # Multi-head attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False) 
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Feed-forward network
        self.gate_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)

        # Layer norms
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Layer norms
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape

        # Self-attention
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, value_states)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attention_output = self.o_proj(attention_output)

        # Add residual
        hidden_states = residual + self.dropout(attention_output)

        # Feed-forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        gate_output = F.silu(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)
        intermediate_output = gate_output * up_output
        hidden_states = self.down_proj(intermediate_output)

        # Add residual
        hidden_states = residual + self.dropout(hidden_states)

        return hidden_states

class ExpertChoiceMoRLayer(nn.Module):
    """Expert-choice MoR layer - experts choose which tokens to process."""

    def __init__(self, hidden_size: int, num_recursions: int = 3, num_heads: int = 8, 
                 capacity_factor: float = 1.0, router_type: str = "linear", alpha: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_recursions = num_recursions
        self.capacity_factor = capacity_factor
        self.alpha = alpha

        # Shared recursion blocks
        self.recursion_blocks = nn.ModuleList([
            SimpleTransformerBlock(hidden_size, num_heads, dropout=dropout) 
            for _ in range(num_recursions)
        ])

        # Routers for each recursion step
        if router_type == "linear":
            self.routers = nn.ModuleList([
                LinearRouter(hidden_size, num_experts=1) 
                for _ in range(num_recursions)
            ])
        else:
            self.routers = nn.ModuleList([
                MLPRouter(hidden_size, num_experts=1) 
                for _ in range(num_recursions)
            ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        total_output = hidden_states.clone()
        all_router_logits = []
        total_load_balancing_loss = 0.0

        # Process through each recursion step
        for step, (block, router) in enumerate(zip(self.recursion_blocks, self.routers)):
            # Compute routing scores
            router_logits = router(hidden_states)  # (batch_size, seq_len, 1)
            router_probs = torch.sigmoid(router_logits) * self.alpha
            all_router_logits.append(router_logits)

            top_k = max(1, int(self.capacity_factor * seq_len))
            weights, selected_indices = torch.topk(router_probs.squeeze(-1), top_k, dim=-1, sorted=False)
            selected_indices, sort_indices = torch.sort(selected_indices, dim=-1)
            weights = torch.gather(weights, dim=-1, index=sort_indices)

            selected_indices_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
            selected_tokens = torch.gather(hidden_states, dim=1, index=selected_indices_expanded)

            # Create attention mask for selected tokens
            if attention_mask is not None:
                selected_seq_len = selected_tokens.size(1)
                selected_attention_mask = torch.triu(torch.ones(selected_seq_len, selected_seq_len, device=hidden_states.device), diagonal=1)
                selected_attention_mask = selected_attention_mask.masked_fill(selected_attention_mask == 1, float('-inf')).unsqueeze(0).unsqueeze(0)
            else:
                selected_attention_mask = None

            # Process through recursion block
            processed_tokens = block(selected_tokens, attention_mask=selected_attention_mask)

            # Weight the processed tokens
            weighted_tokens = processed_tokens * weights.unsqueeze(-1)

            # Scatter back to original positions
            total_output.scatter_add_(dim=1, index=selected_indices_expanded, src=weighted_tokens)

            # Compute load balancing loss
            if self.training:
                expert_usage = router_probs.mean()  # Mean usage across batch and sequence
                load_balancing_loss = expert_usage * (1 - expert_usage)  # Encourage 0.5 usage
                total_load_balancing_loss += load_balancing_loss

        return MoROutput(
            hidden_states=total_output,
            router_logits=torch.stack(all_router_logits, dim=0),
            load_balancing_loss=total_load_balancing_loss / len(self.recursion_blocks) if self.training else None
        )

class TokenChoiceMoRLayer(nn.Module):
    """Token-choice MoR layer - tokens choose their recursion depth."""

    def __init__(self, hidden_size: int, num_recursions: int = 3, num_heads: int = 8, 
                 router_type: str = "linear", alpha: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_recursions = num_recursions
        self.alpha = alpha

        # Shared recursion blocks
        self.recursion_blocks = nn.ModuleList([
            SimpleTransformerBlock(hidden_size, num_heads, dropout=dropout) 
            for _ in range(num_recursions)
        ])

        # Single router that assigns recursion depth to each token
        if router_type == "linear":
            self.router = LinearRouter(hidden_size, num_experts=num_recursions)
        else:
            self.router = MLPRouter(hidden_size, num_experts=num_recursions)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Get routing decisions for all tokens
        router_logits = self.router(hidden_states) 
        router_probs = F.softmax(router_logits, dim=-1) * self.alpha

        # Token-choice: each token gets assigned to one recursion depth
        _, token_assignments = torch.topk(router_probs, 1, dim=-1)  
        token_assignments = token_assignments.squeeze(-1) 

        # Get routing weights
        routing_weights = torch.gather(router_probs, dim=-1, index=token_assignments.unsqueeze(-1))

        final_output = hidden_states.clone()
        load_balancing_loss = 0.0

        # Process tokens through their assigned recursion depths
        for depth in range(self.num_recursions):
            # Find tokens assigned to this depth
            depth_mask = (token_assignments == depth)  
            if not depth_mask.any():
                continue

            # Process through all blocks up to assigned depth
            current_hidden = hidden_states.clone()
            for block_idx in range(depth + 1):
                current_hidden = self.recursion_blocks[block_idx](current_hidden, attention_mask)

            # Update only the tokens assigned to this depth
            depth_mask_expanded = depth_mask.unsqueeze(-1).expand(-1, -1, hidden_dim)
            depth_weights = routing_weights.squeeze(-1).unsqueeze(-1).expand(-1, -1, hidden_dim)

            # Weighted update for tokens at this depth
            final_output = torch.where(depth_mask_expanded, 
                                     final_output + current_hidden * depth_weights, 
                                     final_output)

        # Compute load balancing loss
        if self.training:
            depth_distribution = router_probs.mean(dim=(0, 1))  # (num_recursions,)
            uniform_target = torch.ones_like(depth_distribution) / self.num_recursions
            load_balancing_loss = F.kl_div(depth_distribution.log(), uniform_target, reduction='sum')

        return MoROutput(
            hidden_states=final_output,
            router_logits=router_logits,
            selected_tokens=token_assignments,
            routing_weights=routing_weights,
            load_balancing_loss=load_balancing_loss if self.training else None
        )

class SimpleMoRModel(nn.Module):


    def __init__(self, 
                 vocab_size: int = 1000,
                 hidden_size: int = 128, 
                 num_classes: int = 4,
                 num_layers: int = 4,
                 num_recursions: int = 3,
                 num_heads: int = 8,
                 max_seq_len: int = 64,
                 mor_type: str = "expert_choice", #or token_choice
                 capacity_factor: float = 0.8,
                 alpha: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()

        # Create config from parameters
        self.config = MoRConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            num_recursions=num_recursions,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            mor_type=mor_type,
            capacity_factor=capacity_factor,
            alpha=alpha,
            dropout=dropout
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_recursions = num_recursions
        self.mor_type = mor_type

        # Model components
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)

        # MoR layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if mor_type == "expert_choice":
                layer = ExpertChoiceMoRLayer(
                    hidden_size, num_recursions, num_heads, capacity_factor, "linear", alpha, dropout
                )
            else:  # token_choice
                layer = TokenChoiceMoRLayer(
                    hidden_size, num_recursions, num_heads, "linear", alpha, dropout
                )
            self.layers.append(layer)

        # Output layers
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_attention_mask(self, input_ids, seq_len):
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(self, input_ids, labels=None, return_routing_info=False):
        """
        Forward pass through the MoR model.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            labels: Optional labels for training
            return_routing_info: Whether to return routing statistics

        Returns:
            logits: Classification logits (batch_size, num_classes)
            routing_info: Optional routing statistics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_embeds = self.token_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        hidden_states = token_embeds + pos_embeds
        hidden_states = self.dropout(hidden_states)

        # Create attention mask
        attention_mask = self.create_attention_mask(input_ids, seq_len)

        # Process through MoR layers
        total_load_balancing_loss = 0.0
        all_router_outputs = []

        for layer in self.layers:
            output = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = output.hidden_states
            all_router_outputs.append(output)

            if output.load_balancing_loss is not None:
                total_load_balancing_loss += output.load_balancing_loss

        # Final processing
        hidden_states = self.final_norm(hidden_states)

        # Pool for classification (use last token)
        pooled_output = hidden_states[:, -1, :] 
        logits = self.classifier(pooled_output)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

            # Add load balancing loss
            if total_load_balancing_loss > 0:
                loss = loss + 0.01 * total_load_balancing_loss  

        if return_routing_info:
            return logits, all_router_outputs

        if labels is not None:
            return {"logits": logits, "loss": loss, "load_balancing_loss": total_load_balancing_loss}

        return logits

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_routing_analysis(self, input_ids, attention_mask=None):
        """
        Get detailed routing analysis for the input.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Dictionary with routing statistics and analysis
        """
        with torch.no_grad():
            logits, routing_info = self.forward(input_ids, return_routing_info=True)

            # Calculate average entropy across layers
            total_entropy = 0
            entropy_count = 0

            for info in routing_info:
                if hasattr(info, 'router_logits') and info.router_logits is not None:
                    # Calculate entropy from router logits
                    router_probs = F.softmax(info.router_logits, dim=-1)
                    entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10), dim=-1).mean()
                    total_entropy += entropy
                    entropy_count += 1

            avg_entropy = total_entropy / entropy_count if entropy_count > 0 else 0

            analysis = {
                'logits': logits,
                'layer_routing': routing_info,
                'avg_entropy': avg_entropy,
                'total_parameters': self.get_num_parameters()
            }

            return analysis
