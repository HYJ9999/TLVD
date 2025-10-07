import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from src.modules.llm.llm_wrapper import ImprovedLLMWrapper

class TransformerBlock(nn.Module):
    """Self-attention based transformer block."""
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        feedforward = self.ff(x)
        return self.norm2(x + self.dropout(feedforward))

class BeliefTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, belief_dim: int, 
                 n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        # Ensure hidden_dim is divisible by n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        # Input embedding with position encoding
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Gradient checkpointing flag
        self.use_checkpointing = True
        
        # Transformer layers with memory efficient attention
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                n_heads=n_heads,
                ff_dim=hidden_dim * 4,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.size(0)
        
        # Add positional encoding
        x = self.input_embed(x)
        x = self.pos_encoder(x)
        
        # Create attention mask for padding
        mask = self._create_attention_mask(x)
        
        # Apply transformer layers with gradient checkpointing
        for layer in self.transformer_layers:
            if self.use_checkpointing and self.training:
                x = checkpoint.checkpoint(layer, x, mask)
            else:
                x = layer(x, mask)
                
        # Special handling for batch_size=1
        if batch_size == 1:
            x = x.squeeze(0)
            
        # Generate belief state and parameters
        belief_state = self.belief_net(x)
        params = self.param_net(belief_state)
        
        # Scale parameters appropriately
        temperature, top_p = self._scale_parameters(params)
        
        return belief_state, (temperature, top_p)
        
    def _create_attention_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create attention mask for padding tokens."""
        mask = torch.ones(x.size(0), x.size(1), x.size(1), device=x.device)
        return mask.masked_fill(mask == 0, float('-inf'))
        
    def _scale_parameters(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale parameters to appropriate ranges with numerical stability."""
        temperature = torch.clamp(
            torch.sigmoid(params[..., 0:1]) * 0.9 + 0.1,
            min=0.1,
            max=1.0
        )
        top_p = torch.clamp(
            torch.sigmoid(params[..., 1:2]) * 0.8 + 0.1,
            min=0.1,
            max=0.9
        )
        return temperature, top_p

class LLMTransformerAgent(nn.Module):
    """LLM-based Transformer Agent with belief network integration."""
    def __init__(self, input_shape: int, args: Any):
        super().__init__()
        
        # Initialize Transformer-based belief network
        self.belief_transformer = BeliefTransformer(
            input_dim=input_shape,
            hidden_dim=args.hidden_dim,
            belief_dim=args.belief_dim,
            n_heads=args.attention_heads,
            n_layers=args.transformer_layers,
            dropout=args.dropout
        )
        
        # Initialize LLM wrapper
        self.llm_wrapper = ImprovedLLMWrapper(
            api_key=args.together_api_key,
            model_name=args.executor_model,
            belief_dim=args.belief_dim
        )
        
        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(args.belief_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.n_actions)
        )
        
        # Cache for current parameters
        self.current_params = {'temperature': 0.7, 'top_p': 0.9}
        
    def forward(self, inputs: torch.Tensor, test_mode: bool = False) -> Dict[str, torch.Tensor]:
        # Generate belief state and LLM parameters
        belief_state, llm_params = self.belief_transformer(inputs)
        
        # Generate action logits
        action_out = self.output_net(belief_state)
        
        if test_mode:
            temperature, top_p = 0.7, 0.9
        else:
            temperature, top_p = llm_params
        
        # Cache parameters
        self.current_params = {
            'temperature': temperature,
            'top_p': top_p
        }
        
        return {
            'action_out': action_out,
            'belief_state': belief_state,
            'temperature': temperature,
            'top_p': top_p
        }
    
    def generate_answer(self, question: str, strategy: str, 
                       belief_state: Optional[torch.Tensor] = None) -> str:
        # Use belief state to adjust LLM parameters
        temperature = self.current_params['temperature']
        top_p = self.current_params['top_p']
        
        prompt = f"""You are an execution LLM in a multi-agent system.
Question: {question}
Strategy: {strategy}
Your task is to provide a solution based on the strategy.
Please ensure your answer is clear and follows the strategy.

Answer:"""
        
        return self.llm_wrapper.generate_response(
            prompt=prompt,
            belief_state=belief_state,
            temperature=temperature,
            top_p=top_p
        )
        
    def init_hidden(self) -> torch.Tensor:
        # No hidden state needed for transformer
        return torch.zeros(1)