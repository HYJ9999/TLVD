import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math
from modules.llm.llm_wrapper import ImprovedLLMWrapper
import logging

logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PositionalEncoding(nn.Module):
    """
    Positional encoding layer providing positional information for Transformer inputs.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Directly use the registered positional encoding buffer to avoid reassignment
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """
    Self-attention Transformer block, basic building block.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True).to(device)
        self.norm1 = nn.LayerNorm(embed_dim).to(device)
        self.norm2 = nn.LayerNorm(embed_dim).to(device)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        ).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention layer
        # Handle attention mask: PyTorch MultiheadAttention expects a 2D mask of shape (seq_len, seq_len)
        attn_mask = None
        if mask is not None:
            if mask.ndim == 2 and mask.shape[0] == x.shape[0]:  # (batch_size, seq_len)
                # Convert to (seq_len, seq_len), same for every batch
                seq_len = mask.shape[1]
                # Here we assume mask is a padding mask, where True indicates positions to ignore
                # For self-attention, we need a mask of shape (seq_len, seq_len)
                # Simple approach: if a position is masked in any batch, mask it for all positions
                # Or we can directly use the mask from the first batch
                position_mask = mask[0]  # Use the first batch's mask (seq_len,)
                # Create a (seq_len, seq_len) mask
                attn_mask = position_mask.unsqueeze(0).expand(seq_len, -1) | position_mask.unsqueeze(1).expand(-1, seq_len)
            elif mask.ndim == 2 and mask.shape[0] == mask.shape[1]:  # Already (seq_len, seq_len)
                attn_mask = mask
        attended, _ = self.attention(x, x, x, attn_mask=attn_mask)
        
        # Residual connection and layer normalization - avoid in-place operations
        residual_1 = x + attended
        attended = self.norm1(residual_1)
        
        # Feed-forward network
        ff_output = self.feed_forward(attended)
        
        # Residual connection and layer normalization - avoid in-place operations
        residual_2 = attended + ff_output
        output = self.norm2(residual_2)

        
        return output

class BeliefNetwork(nn.Module):
    """
    Individual belief network B_i, used to maintain and update the agent's belief state b_i.
    According to the ECON paper, this network receives the local trajectory τ_i^t and current observation o_i^t,
    and outputs belief state b_i, prompt embedding e_i = [T_i, p_i], and local Q-value Q_i^t.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, belief_dim: int, 
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 T_min: float = 0.1, T_max: float = 2.0, p_min: float = 0.1, p_max: float = 0.9,
                 vocab_size: int = 50257):  # Add vocabulary size parameter, GPT2's default
        super(BeliefNetwork, self).__init__()
        
        # Save parameters
        self.observation_dim = observation_dim  # This is max_token_length
        self.belief_dim = belief_dim
        self.T_min = T_min
        self.T_max = T_max
        self.p_min = p_min
        self.p_max = p_max
        
        # Token embedding layer: convert token IDs to dense vectors
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                ff_dim=hidden_dim * 4, # Standard practice for ff_dim 4
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Output mapping layer (from hidden_dim to belief_dim)
        self.belief_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # Added a layer for more capacity
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, belief_dim)
        )
        
        # Prompt embedding parameter generation network (corresponding to W_T, b_T, W_p, b_p)
        # Create separate linear layers for T and p, closer to the paper's formula
        self.temp_projection = nn.Linear(belief_dim, 1) # W_T b_i + b_T
        self.penalty_projection = nn.Linear(belief_dim, 1) # W_p b_i + b_p
        
        # Q-value prediction network (parameters φ_i)
        self.q_network = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 1)  # Output scalar Q value
        )
        
    def forward(self, token_ids: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass: generate belief state, prompt embedding, and Q values from token IDs.
        
        Args:
            token_ids: Token IDs tensor with shape (batch_size, seq_len) or (batch_size, 1, seq_len)
            mask: Optional attention mask
            
        Returns:
            A dictionary containing belief state b_i, scaled prompt embedding e_i = [T_i, p_i], and local Q-value Q_i^t.
        """
        # Ensure token_ids have the correct shape
        if token_ids.ndim == 1:  # Output tensor dimension
            token_ids = token_ids.unsqueeze(0)  # (seq_len,) -> (1, seq_len)
        if token_ids.ndim == 3:
            # Avoid in-place operations; use view instead of squeeze
            batch_size, _, seq_len = token_ids.shape
            token_ids = token_ids.view(batch_size, seq_len)  # (batch, 1, seq_len) -> (batch, seq_len)
        
        # Token embedding
        x = self.token_embedding(token_ids.long())  # (batch_size, seq_len, hidden_dim)
        
        # Positional encoding
        x = self.pos_encoder(x) # x is (batch, seq_len, hidden_dim)
        
        # Apply Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask) # x is (batch, seq_len, hidden_dim)


        # Safely take the last valid token of each sequence (use gather instead of advanced indexing). Why take the last valid token here?
        batch_size, seq_len, hidden_dim = x.size()
        if mask is not None:
            # mask: True indicates positions to ignore (padding). Compute valid length for each sample
            valid_lengths = (~mask).sum(dim=1)  # (batch,)
            last_idx = (valid_lengths - 1).clamp(min=0)  # (batch,)
            # expand for gather: need shape (batch, 1, hidden_dim)
            indices = last_idx.view(batch_size, 1, 1).expand(-1, 1, hidden_dim)  # (batch,1,hidden_dim)
            gathered_sequence = x.gather(1, indices)  # (batch, 1, hidden_dim)
            processed_sequence = gathered_sequence.contiguous().view(batch_size, hidden_dim)  # Ensure memory continuity; avoid in-place operations
        else:
            processed_sequence = x[:, -1, :].clone()  # (batch, hidden_dim) — use clone to avoid shared memory

        # Generate belief state b_i
        belief_state = self.belief_projection(processed_sequence) # (batch, belief_dim)
        
        # Generate prompt embedding e_i = [T_i, p_i]
        # T_i = T_min + (T_max - T_min) * σ(W_T b_i + b_T)
        # p_i = p_min + (p_max - p_min) * σ(W_p b_i + b_p)
        
        temp_logit = self.temp_projection(belief_state) # (batch, 1) MLP layer
        penalty_logit = self.penalty_projection(belief_state) # (batch, 1) MLP layer
        
        temperature = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(temp_logit)
        penalty = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(penalty_logit)
        
        # prompt_embedding_scaled shape: (batch_size, 2)
        prompt_embedding_scaled = torch.cat([temperature, penalty], dim=1)
        
        # Generate Q value Q_i^t, input is belief_state
        q_value = self.q_network(belief_state) # (batch, 1) MLP layer
        
        return {
            'belief_state': belief_state,          # b_i
            'prompt_embedding': prompt_embedding_scaled, # e_i = [T_i, p_i]
            'q_value': q_value,                    # Q_i^t
            'temp_logit': temp_logit,              # Original temperature logit
            'penalty_logit': penalty_logit         # Original repetition penalty logit
        }

class LLMTransformerAgent(nn.Module):
    """
    Transformer-based LLM agent that maintains belief state and generates dynamic prompt embeddings.
    """
    def __init__(self, input_shape: int, args: Any): # input_shape now represents observation_dim + action_dim or just observation_dim
        super(LLMTransformerAgent, self).__init__()
        
        # Parameter settings
        self.args = args
        # self.input_shape = input_shape # Changed to directly use dimensions from args or infer
        self.belief_dim = args.belief_dim
        # Correctly access use_cuda property
        use_cuda = hasattr(args, 'system') and hasattr(args.system, 'use_cuda') and args.system.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        
        # Temperature and top_p ranges (from args or BeliefNetwork defaults)
        self.T_min = getattr(args.sampling, 'temperature_min', 0.1)
        self.T_max = getattr(args.sampling, 'temperature_max', 2.0)
        self.p_min = getattr(args.sampling, 'p_min', 0.1)
        self.p_max = getattr(args.sampling, 'p_max', 0.9)
        
        # Initialize individual belief network
        # Input should be the length of tokenized observation, not state_shape
        # Get maximum token length from config
        max_token_len = getattr(args.env_args, "max_question_length", 512)
        belief_network_input_dim = max_token_len  # Use tokenized observation length

        self.belief_network = BeliefNetwork(
            observation_dim=belief_network_input_dim, # Explicitly pass observation dimension
            action_dim=0, # Assume actions are included in observations or not used directly as independent inputs for BeliefNetwork base embedding layer
            hidden_dim=getattr(args.arch, 'entity_dim', 256),
            belief_dim=self.belief_dim,
            n_heads=getattr(args.arch, 'attention_heads', 4),
            n_layers=getattr(args.arch, 'transformer_blocks', 2),
            dropout=getattr(args.arch, 'dropout_rate', 0.1),
            T_min=self.T_min, T_max=self.T_max, 
            p_min=self.p_min, p_max=self.p_max,
            vocab_size=getattr(args, 'vocab_size', 50257)  # Add vocabulary size
        ).to(device)
        
        # Output layer (generate action probabilities) - this may need to be revisited
        # In ECON, actions are the prompt_embedding e_i. Here, output_network may be for a discrete action space.
        # If the framework fully relies on e_i as actions, this network may be unnecessary or used differently.
        # Keep it just in case, but mark that it needs confirmation according to the overall ECON action selection mechanism.
        self.output_network = nn.Sequential(
            nn.Linear(self.belief_dim, args.n_actions)
        ).to(device)
        
        # Initialize LLM wrapper
        self.llm_wrapper = ImprovedLLMWrapper(
            api_key=args.together_api_key,
            model_name=args.executor_model,
            belief_dim=self.belief_dim # LLM Wrapper may also need belief state
        )
        
        # Cache latest prompt embedding
        self.current_prompt_embedding_tensor = torch.tensor([ (self.T_min + self.T_max) / 2, (self.p_min + self.p_max) / 2 ], device=self.device) # (2,)
        
        # Initialize prompt embedding dictionary (for logging and debugging)
        self.current_prompt_embedding = {
            'temperature': (self.T_min + self.T_MAX) / 2 if hasattr(self, 'T_MAX') else (self.T_min + self.T_max) / 2,
            'repetition_penalty': (self.p_min + self.p_max) / 2
        }
        
    def forward(self, inputs: torch.Tensor, hidden_state: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None, test_mode: bool = False) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs the forward pass, generating action probabilities, the belief state, and the prompt embedding.

        Args:
            inputs (torch.Tensor): The input tensor representing $\tau_i^t$ and $o_i^t$.
            hidden_state (torch.Tensor, optional): The optional hidden state (not used in the Transformer).
            mask (torch.Tensor, optional): The optional attention mask.
            test_mode (bool): Whether the system is operating in test mode.

        Returns:
            Tuple[dict, Any]: 
                - A dictionary containing various outputs.
                - The new hidden state (which is practically unused).
        """
        # Obtain the belief state, prompt embedding e_i, and local Q-value Q_i^t via the belief network.
        # The 'inputs' parameter is now understood to be local_history_obs.
        belief_outputs = self.belief_network(inputs, mask)
        
        belief_state = belief_outputs['belief_state']
        prompt_embedding = belief_outputs['prompt_embedding'] # e_i
        local_q_value = belief_outputs['q_value'] # Q_i^t
        temp_logit = belief_outputs['temp_logit'] # Original Temperature logit
        penalty_logit = belief_outputs['penalty_logit'] # Original Penalty logit
        
        # Update the cached prompt embedding (tensor form)
        if not test_mode:
            # In training mode, use the first sample of the network-generated prompt_embedding (if batch_size > 1).
            # Alternatively, if batch_size is always 1, just use it directly.
            self.current_prompt_embedding_tensor = prompt_embedding[0].detach().clone() if prompt_embedding.shape[0] > 0 else self.current_prompt_embedding_tensor


        # The action in ECON is the prompt_embedding e_i.
        # The role of the output_network needs to be determined based on the specific scenario.
        # Retain it if it is used to derive other types of actions from the belief_state (e.g., discrete action selection).
        # If the action in the ECON framework is solely e_i, then action_q_values might not be directly used for final action selection,
        # or the Q_i^t value itself might represent the value estimation for e_i.
        action_q_values = self.output_network(belief_state) # This may be for auxiliary tasks or a different action space
        
        # In test mode, you can override the generated prompt_embedding
        if test_mode:
            # Use default or pre-set test parameters.
            # Note: Previously, we used fixed values (0.7, 0.9). Whether to override
            # depends on the testing strategy if the BeliefNetwork should run during testing.
            # To maintain consistency with previous behavior, we can set fixed test values here.
            # Ideally, however, 'test_mode' should allow the network to generate its own values, unless an override is explicitly required.
            test_temp = torch.tensor([(self.T_min + self.T_max) / 2], device=self.device)
            test_penalty = torch.tensor([(self.p_min + self.p_max) / 2], device=self.device)
            prompt_embedding = torch.cat([test_temp, test_penalty], dim=0).unsqueeze(0) # (1, 2)
            if belief_state.shape[0] > 1: # If this is a batch input
                prompt_embedding = prompt_embedding.repeat(belief_state.shape[0], 1)


        outputs = {
            "action_q_values": action_q_values, 
            "belief_state": belief_state,       # b_i
            "prompt_embedding": prompt_embedding, # e_i = [T_i, p_i]
            "q_value": local_q_value,           # Q_i^t - Maintain field names consistent with the BeliefNetwork
            "raw_prompt_embed_params": torch.cat([temp_logit, penalty_logit], dim=1) # Save original logits for potential analysis or loss computation
        }
        
    # TransformerAgent usually returns (outputs_dict, hidden_state).
    # The hidden_state is not required here for the Transformer-based BeliefNetwork, so we return None or the belief_state.
        return outputs, belief_state # Or (outputs, None)
    
    def generate_answer(self, question: str, strategy:Optional[str]=None, 
                       belief_state: Optional[torch.Tensor] = None, # Optional because the agent generates it internally
                       temperature: Optional[float] = None, # Will be obtained from self.current_prompt_embedding_tensor
                       repetition_penalty: Optional[float] = None) -> str: # Corresponds to p_i in the paper
        """
        Uses the LLM to generate latent variable descriptions, based on the current belief state and dynamically generated prompt embeddings.

        Args:
            question (str): The input question, which includes the latent variable description.
            strategy (str): A customized strategy used to guide the LLM's generation.
            belief_state (dict, optional): The current belief state. If not provided, the agent will attempt to use its internal state.
            temperature (float, optional): Overrides the dynamically generated temperature.
            repetition_penalty (float, optional): Overrides the dynamically generated repetition penalty ($p_i$).

        Returns:
            str: The latent variable description string generated by the LLM.
        """
        
        # Retrieve the current prompt embedding parameters
        # self.current_prompt_embedding_tensor stores [T_i, p_i]
        current_temp = self.current_prompt_embedding_tensor[0].item()
        current_penalty = self.current_prompt_embedding_tensor[1].item()

        final_temp = temperature if temperature is not None else current_temp
        final_penalty = repetition_penalty if repetition_penalty is not None else current_penalty
        
        # Update the current_prompt_embedding dictionary, primarily for logging or debugging; the actual parameters are passed to the LLM
        self.current_prompt_embedding['temperature'] = final_temp
        
        self.current_prompt_embedding['repetition_penalty'] = final_penalty

        system_prompt = """You are a causal inference engine.
Task:
You will be given:
- A latent variable for which you need to infer the description.
- A JSON dictionary containing descriptions of **other latent variables** (ignore the description of the specified latent variable) and **known observed variables**.
- A causal graph that represents the relationships between latent and observed variables.

Your job:
Based on the given latent variable and the provided context (descriptions of other latent variables and known observed variables), infer **the most plausible, coherent, and concise** description of the **specified latent variable**.
- The description should reflect the underlying mechanism of the latent variable, i.e., how it can reasonably influence all of the observed variables it is associated with.
- The description should be in the form of **a short phrase**, as short as possible, without unnecessary elaboration.
- Ensure that the description of the latent variable is **distinct** and does not closely resemble the descriptions of any other latent or observed variables.
- The description should be clear, concise, and coherent.

Input format:
- Latent variable: A specific latent variable (e.g., L1, L2, ...).
- Latent_var_desc: A JSON dictionary containing descriptions of other latent variables (including the description of the specified latent variable, but ignore it while inferring the description).
- Known_var_desc: A JSON dictionary of known observed variables and their descriptions.
- Causal graph: A graph representation (in JSON or any other format) that shows the causal relationships between latent and observed variables. This graph will be **key** in understanding how the latent variable impacts the observed variables.

Constraints:
- The description should be the most plausible and unique for the given latent variables and the known observed variables.
- Keep the description **concise**, focusing on key mechanisms.
- **Ignore** the description of the specified latent variable in Latent_var_desc **if present**.
- Do not include any **commentary, explanations, or reasoning** in the output.
- Must include the confidence score in the output.
- The confidence score should be between 0 and 1, with higher values indicating higher confidence in the description.

Output format:
Only return a JSON dictionary mapping the specified latent variable to its description and a confidence score,do not include any other text.

Example:
{
  "L1": "General predisposition to pain sensitivity based on genetic and physiological factors.",
  "confidence": 0.95
}
"""



        answer = self.llm_wrapper.generate_response(
            prompt=question,
            strategy=system_prompt,  # Strategy is already included in the prompt
            temperature=final_temp,
            repetition_penalty=final_penalty,
            max_tokens=400  # Reasonable length for mathematical solutions
        )
        
        
        return answer
        
    def save_models(self, path: str):
        """
        Saves the model parameters.

        Args:
            path (str): The saving path.
        """
        torch.save(self.belief_network.state_dict(), f"{path}/belief_network.th")
        torch.save(self.output_network.state_dict(), f"{path}/output_network.th")
    
    def load_models(self, path: str):
        """
        Loads the model parameters.

        Args:
            path (str): The loading path.
        """
        self.belief_network.load_state_dict(torch.load(f"{path}/belief_network.th"))
        self.output_network.load_state_dict(torch.load(f"{path}/output_network.th"))
    
    def cuda(self):
        """
        Moves the model parameters to the CUDA device.
        """
        self.belief_network.cuda()
        self.output_network.cuda()
        
    def init_hidden(self):
        """
        Initializes the hidden state (not used in the Transformer, but retained for interface compatibility).
        """
        
        return torch.zeros(1, device=self.device)