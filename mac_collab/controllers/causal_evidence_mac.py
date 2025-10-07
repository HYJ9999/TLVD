import torch
import torch.nn as nn
from modules.agents.transformer_agent import LLMTransformerAgent
from modules.belief_encoder import BeliefEncoder
from components.action_selectors import REGISTRY as action_REGISTRY
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

class CausalEvidenceMAC:
    """
    Multi-agent controller for causal evidence search
    Coordinate behaviors of the paper agent and the Wikipedia agent
    """
    def __init__(self, scheme: Dict, groups: Dict, args: Any):
        self.n_agents = 2  # Paper agent and Wikipedia agent
        self.args = args
        
        # Device configuration
        use_cuda = hasattr(args, 'system') and hasattr(args.system, 'use_cuda') and args.system.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        # Get input dimensions
        self.input_shape = scheme["state"]["vshape"]
        
        # Initialize agents
        self._build_agents()
        
        # Action selector
        self.action_selector = action_REGISTRY[args.action_selector](args)
        
        # Initialize belief encoder
        self.belief_encoder = BeliefEncoder(
            belief_dim=args.belief_dim,
            n_agents=self.n_agents,
            n_heads=args.belief_n_heads,
            key_dim=args.belief_hidden_dim,
            device=self.device
        )
        
        # Agent states
        self.paper_agent_state = {
            "current_keywords": None,  # Current keywords in use
            "papers_found": 0,  # Number of related papers found
            "evidence_strength": 0.0,  # Evidence strength (0-1)
            "belief_state": torch.zeros(args.belief_dim, device=self.device),  # Belief state
            "prompt_embedding": torch.zeros(2, device=self.device),  # [task_id, prompt_id]
        }
        
        self.wiki_agent_state = {
            "current_keywords": None,  # Current keywords in use
            "evidence_found": False,  # Whether evidence is found
            "evidence_strength": 0.0,  # Evidence strength (0-1)
            "belief_state": torch.zeros(args.belief_dim, device=self.device),  # Belief state
            "prompt_embedding": torch.zeros(2, device=self.device),  # [task_id, prompt_id]
        }
        
    def _build_agents(self):
        """Initialize agents"""
        try:
            # Paper agent and Wikipedia agent use independent network architectures
            self.paper_agent = LLMTransformerAgent(
                input_shape=self.input_shape,
                args=self.args,
                device=self.device
            )
            
            self.wiki_agent = LLMTransformerAgent(
                input_shape=self.input_shape,
                args=self.args,
                device=self.device
            )
            
            self.agents = [self.paper_agent, self.wiki_agent]
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise
            
    def init_hidden(self, batch_size: int):
        """Initialize hidden states"""
        # Reset agent states
        self.paper_agent_state["belief_state"] = torch.zeros(
            batch_size, self.args.belief_dim,
            device=self.device
        )
        self.paper_agent_state["prompt_embedding"] = torch.zeros(
            batch_size, 2,
            device=self.device
        )
        
        self.wiki_agent_state["belief_state"] = torch.zeros(
            batch_size, self.args.belief_dim,
            device=self.device
        )
        self.wiki_agent_state["prompt_embedding"] = torch.zeros(
            batch_size, 2,
            device=self.device
        )
        
    def forward(self, ep_batch: Any, t: int, test_mode: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass to generate actions
        
        Args:
            ep_batch: experience batch
            t: current time step
            test_mode: whether in test mode
            
        Returns:
            agent_outputs: outputs of agents' actions
            extra_info: additional information
        """
        agent_inputs = self._build_inputs(ep_batch, t)
        
        # Get belief state of each agent
        paper_belief = self.paper_agent_state["belief_state"]
        wiki_belief = self.wiki_agent_state["belief_state"]
        belief_states = torch.stack([paper_belief, wiki_belief], dim=1)
        
        # Use belief encoder to generate group representation
        group_repr = self.belief_encoder(belief_states)
        
        # Get each agent's actions
        agent_outs = []
        for i in range(self.n_agents):
            agent_out = self.agents[i](agent_inputs[:, i], group_repr)
            agent_outs.append(agent_out)
            
        agent_outs = torch.stack(agent_outs, dim=1)
        
        # Use greedy strategy in test mode
        if test_mode:
            agent_outs = agent_outs.max(dim=-1)[1]
        
        return agent_outs, {"group_repr": group_repr}
        
    def _build_inputs(self, batch: Any, t: int) -> torch.Tensor:
        """
        Build agent inputs
        
        Args:
            batch: experience batch
            t: current time step
            
        Returns:
            inputs: input tensor for agents
        """
        bs = batch.batch_size
        
        # Get states
        inputs = []
        for i in range(self.n_agents):
            agent_state = batch["state"][:, t].reshape(bs, -1)
            if i == 0:  # Paper agent
                agent_state = torch.cat([
                    agent_state,
                    self.paper_agent_state["belief_state"],
                    self.paper_agent_state["prompt_embedding"]
                ], dim=-1)
            else:  # Wikipedia agent
                agent_state = torch.cat([
                    agent_state,
                    self.wiki_agent_state["belief_state"],
                    self.wiki_agent_state["prompt_embedding"]
                ], dim=-1)
            inputs.append(agent_state)
            
        return torch.stack(inputs, dim=1)  # [bs, n_agents, input_dim]
        
    def update_belief_states(self, paper_evidence: float, wiki_evidence: float):
        """
        Update belief states
        
        Args:
            paper_evidence: paper evidence strength
            wiki_evidence: Wikipedia evidence strength
        """
        # Update belief state of the paper agent
        if paper_evidence > 0:
            update_mask = torch.rand_like(self.paper_agent_state["belief_state"]) < paper_evidence
            self.paper_agent_state["belief_state"][update_mask] += (
                torch.randn_like(self.paper_agent_state["belief_state"][update_mask]) * paper_evidence
            )
        else:
            self.paper_agent_state["belief_state"] *= 0.95
            
        # Update belief state of the Wikipedia agent
        if wiki_evidence > 0:
            update_mask = torch.rand_like(self.wiki_agent_state["belief_state"]) < wiki_evidence
            self.wiki_agent_state["belief_state"][update_mask] += (
                torch.randn_like(self.wiki_agent_state["belief_state"][update_mask]) * wiki_evidence
            )
        else:
            self.wiki_agent_state["belief_state"] *= 0.95
            
        # Normalize
        self.paper_agent_state["belief_state"] = torch.nn.functional.normalize(
            self.paper_agent_state["belief_state"], p=2, dim=-1
        )
        self.wiki_agent_state["belief_state"] = torch.nn.functional.normalize(
            self.wiki_agent_state["belief_state"], p=2, dim=-1
        )