# components/action_selectors.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class MultinomialActionSelector:
    """
    Selects actions using a multinomial distribution over q-values.
    Supports epsilon-greedy exploration.
    """
    def __init__(self, args):
        self.args = args
        self.schedule_start = getattr(args, "epsilon_start", 1.0)
        self.schedule_finish = getattr(args, "epsilon_finish", 0.05)
        self.schedule_timesteps = getattr(args, "epsilon_anneal_time", 50000)
        self.epsilon = self.schedule_start
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs: torch.Tensor,
                    avail_actions: torch.Tensor,
                    t_env: int,
                    test_mode: bool = False) -> torch.Tensor:
        """
        Select actions based on multinomial sampling.
        
        Args:
            agent_inputs: Q-values or policy outputs
            avail_actions: Available actions mask
            t_env: Current environment timestep
            test_mode: Whether in testing mode
            
        Returns:
            Selected actions
        """
        masked_q_values = self._mask_actions(agent_inputs, avail_actions)
        
        # Epsilon schedule
        if test_mode and self.test_greedy:
            epsilon = 0.0
        else:
            epsilon = self.epsilon if t_env <= self.schedule_timesteps else self.schedule_finish
            
        # Random exploration
        random_numbers = torch.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < epsilon).long()
        
        # Get random and greedy actions
        random_actions = self._get_random_actions(avail_actions)
        greedy_actions = masked_q_values.max(dim=2)[1]
        
        # Combine random and greedy actions
        chosen_actions = pick_random * random_actions + (1 - pick_random) * greedy_actions
        
        return chosen_actions

    def _mask_actions(self, agent_inputs: torch.Tensor,
                     avail_actions: torch.Tensor) -> torch.Tensor:
        """Apply action masking."""
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = -float("inf")
        return masked_q_values

    def _get_random_actions(self, avail_actions: torch.Tensor) -> torch.Tensor:
        """Sample random available actions."""
        # Replace zeros with very small value to avoid division by zero
        avail_actions_nonzero = avail_actions + 1e-10
        
        # Normalize to create probability distribution
        probs = avail_actions_nonzero / avail_actions_nonzero.sum(dim=-1, keepdim=True)
        
        # Sample from distribution
        return torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[0], -1)

    def epsilon_decay(self, t_env: int):
        """Update epsilon according to schedule."""
        if t_env <= self.schedule_timesteps:
            self.epsilon = self.schedule_start - (self.schedule_start - self.schedule_finish) * (
                t_env / self.schedule_timesteps
            )

class GaussianActionSelector:
    """
    Selects actions using a Gaussian distribution.
    Suitable for continuous action spaces.
    """
    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)
        self.gaussian_std = getattr(args, "gaussian_std", 0.1)
        self.std_decay_rate = getattr(args, "std_decay_rate", 0.99)
        self.min_std = getattr(args, "min_std", 0.02)

    def select_action(self, agent_inputs: torch.Tensor,
                     avail_actions: Optional[torch.Tensor] = None,
                     t_env: Optional[int] = None,
                     test_mode: bool = False) -> torch.Tensor:
        """
        Select actions using Gaussian noise.
        
        Args:
            agent_inputs: Mean actions from policy
            avail_actions: Not used in continuous case
            t_env: Current environment timestep
            test_mode: Whether in testing mode
            
        Returns:
            Selected actions with noise
        """
        if test_mode and self.test_greedy:
            return agent_inputs
            
        # Add Gaussian noise
        noise = torch.randn_like(agent_inputs) * self.gaussian_std
        actions = agent_inputs + noise
        
        # Clip actions if needed
        if hasattr(self.args, "action_range"):
            actions = torch.clamp(actions, 
                                min=self.args.action_range[0],
                                max=self.args.action_range[1])
        
        return actions

    def update_std(self, t_env: int):
        """Decay standard deviation over time."""
        self.gaussian_std = max(
            self.gaussian_std * self.std_decay_rate,
            self.min_std
        )

class EpsilonGreedyActionSelector:
    """
    epsilon-greedy action selector
    """
    def __init__(self, args):
        self.args = args
        
        self.schedule_start = getattr(args, "epsilon_start", 1.0)
        self.schedule_finish = getattr(args, "epsilon_finish", 0.05)
        self.schedule_timesteps = getattr(args, "epsilon_anneal_time", 50000)
        self.schedule_mode = getattr(args, "epsilon_decay_mode", "linear")
        
        self.epsilon = self.schedule_start
        self.test_greedy = getattr(args, "test_greedy", True)
        
    def select_action(self, agent_inputs: torch.Tensor, avail_actions: Optional[torch.Tensor] = None,
                     t_env: Optional[int] = None, test_mode: bool = False) -> torch.Tensor:
        """
        Select an action
        
        Args:
            agent_inputs: Agent inputs, shape (batch_size, n_agents, n_actions)
            avail_actions: Available actions mask, shape (batch_size, n_agents, n_actions)
            t_env: Current environment timestep
            test_mode: Whether in test mode
            
        Returns:
            Selected actions, shape (batch_size, n_agents)
        """
        # Update exploration rate
        if t_env is not None:
            self.epsilon = self.get_epsilon(t_env)
            
        # Use greedy strategy in test mode
        if test_mode and self.test_greedy:
            return self.greedy_action(agent_inputs, avail_actions)
            
        # Get random actions
        batch_size = agent_inputs.size(0)
        n_agents = agent_inputs.size(1)
        n_actions = agent_inputs.size(2)
        
        if avail_actions is not None:
            random_numbers = torch.rand_like(agent_inputs[:, :, 0])  # (batch_size, n_agents)
            random_actions = torch.zeros_like(random_numbers, dtype=torch.long)  # (batch_size, n_agents)
            
            # Randomly choose an available action for each agent
            for b in range(batch_size):
                for a in range(n_agents):
                    avail = avail_actions[b, a]  # (n_actions,)
                    if avail.sum() > 0:  # If there are available actions
                        probs = avail.float() / avail.sum()
                        random_actions[b, a] = torch.multinomial(probs, 1)
        else:
            random_actions = torch.randint(0, n_actions, (batch_size, n_agents))
            
        # Get greedy actions
        greedy_actions = self.greedy_action(agent_inputs, avail_actions)
        
        # Choose actions based on exploration rate
        pick_random = torch.rand_like(random_actions.float()) < self.epsilon
        picked_actions = torch.where(pick_random, random_actions, greedy_actions)
        
        return picked_actions
        
    def greedy_action(self, agent_inputs: torch.Tensor,
                     avail_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Greedy action selection
        
        Args:
            agent_inputs: Agent inputs, shape (batch_size, n_agents, n_actions)
            avail_actions: Available actions mask, shape (batch_size, n_agents, n_actions)
            
        Returns:
            Selected actions, shape (batch_size, n_agents)
        """
        if avail_actions is not None:
            # Set values of unavailable actions to negative infinity
            agent_inputs[avail_actions == 0] = float("-inf")
            
        return agent_inputs.max(dim=-1)[1]  # Return the index of the maximum value
        
    def get_epsilon(self, t_env: int) -> float:
        """
        Get the exploration rate at the current timestep
        
        Args:
            t_env: Current environment timestep
            
        Returns:
            Current exploration rate
        """
        if t_env >= self.schedule_timesteps:
            return self.schedule_finish
            
        if self.schedule_mode == "linear":
            # Linear decay
            epsilon = self.schedule_start - (self.schedule_start - self.schedule_finish) * (
                t_env / self.schedule_timesteps
            )
        elif self.schedule_mode == "exponential":
            # Exponential decay
            epsilon = self.schedule_finish + (self.schedule_start - self.schedule_finish) * np.exp(
                -t_env / (self.schedule_timesteps / 3)
            )
        else:
            raise ValueError(f"Unknown epsilon decay mode: {self.schedule_mode}")
            
        return epsilon

REGISTRY = {
    "multinomial": MultinomialActionSelector,
    "gaussian": GaussianActionSelector,
    "epsilon_greedy": EpsilonGreedyActionSelector
}