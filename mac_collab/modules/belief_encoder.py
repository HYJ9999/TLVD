import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BeliefEncoder(nn.Module):
    """
    Shared belief encoder that aggregates all agents' belief states to produce a group representation.
    
    According to the ECON framework, this module takes in all agents' belief states b_i, processes them
    with multi-head attention, and outputs a group-level representation E.
    """
    
    def __init__(self, belief_dim: int, n_agents: int, n_heads: int = 4, 
                 key_dim: int = 64, device: torch.device = None):
        """
        Initialize the belief encoder.
        
        Args:
            belief_dim: Dimension of belief state
            n_agents: Number of agents
            n_heads: Number of attention heads
            key_dim: Dimension for each attention head
            device: Computation device
        """
        super(BeliefEncoder, self).__init__()
        
        self.belief_dim = belief_dim
        self.n_agents = n_agents
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=belief_dim,
            num_heads=n_heads,
            batch_first=True
        ).to(self.device)
        
        
        self.out_proj = nn.Linear(belief_dim, belief_dim).to(self.device)
        
        
        self.layer_norm = nn.LayerNorm(belief_dim).to(self.device)
        
       
        self.feedforward = nn.Sequential(
            nn.Linear(belief_dim, 4 * belief_dim),
            nn.ReLU(),
            nn.Linear(4 * belief_dim, belief_dim)
        ).to(self.device)
        
       
        self.final_layer_norm = nn.LayerNorm(belief_dim).to(self.device)
        
    def forward(self, belief_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to aggregate agents' belief states and produce a group representation.
        
        Args:
            belief_states: Belief states of all agents [batch_size, n_agents, belief_dim]
            
        Returns:
            Group representation E [batch_size, belief_dim]
        """
        batch_size = belief_states.shape[0]
        
        
        # belief_states: [batch_size, n_agents, belief_dim]
        attn_output, _ = self.multihead_attn(
            query=belief_states,
            key=belief_states,
            value=belief_states
        )
        # attn_output: [batch_size, n_agents, belief_dim]
        
        
        attn_output = belief_states + attn_output
        attn_output = self.layer_norm(attn_output)
        
        
        ff_output = self.feedforward(attn_output)
        
        
        ff_output = attn_output + ff_output
        ff_output = self.final_layer_norm(ff_output)
        
       
        group_repr = ff_output.mean(dim=1)  # [batch_size, belief_dim]
        
        
        group_repr = self.out_proj(group_repr)  # A linear layer
        
        return group_repr
        
    def compute_loss(self, td_loss_tot: torch.Tensor, 
                    td_losses_i: List[torch.Tensor], 
                    lambda_e: float) -> torch.Tensor:

        """
        Calculates the regularization loss (L_e) for the belief encoder.

        According to the ECON framework paper, this loss is:
        L_e(θ_e) = L_TD^tot(φ) + λ_e Σ_i L_TD^i(θ_i^B)

        Args:
            td_loss_tot (float): The global TD loss (L_TD^tot(φ)).
            td_losses_i (List[float]): A list of local TD losses for each individual agent (L_TD^i(θ_i^B)).
            lambda_e (float): The regularization weight for the encoder.
            
        Returns:
            float: The encoder loss (L_e).
        """

        # L_e(θ_e) = L_TD^tot(φ) + λ_e Σ_i L_TD^i(θ_i^B)
        sum_local_td_losses = sum(td_losses_i)
        
        encoder_loss = td_loss_tot + lambda_e * sum_local_td_losses
        
        return encoder_loss