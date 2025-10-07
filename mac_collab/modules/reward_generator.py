# src/modules/reward_controller.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict
import loggers.logger as logger

class RewardController:
    """Controls and adjusts reward weights dynamically."""
    def __init__(self, coordinator_llm, initial_weights=(0.4, 0.4, 0.2)):
        self.coordinator = coordinator_llm
        self.weights = nn.Parameter(torch.tensor(initial_weights))
        self.optimizer = optim.Adam([self.weights], lr=0.001)
        try:
            self.weights = nn.Parameter(torch.tensor(initial_weights))
            self.weights.data = F.softmax(self.weights.data, dim=0)  # Ensure initial weights are normalized
        except Exception as e:
            logger.error(f"Error initializing weights: {e}")
            self.weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.34]))
        
    def get_task_specific_reward(self, answer: str, question: str) -> float:
        """Use LLM to evaluate task-specific performance."""
        prompt = f"""Evaluate this mathematical solution:
Question: {question}
Answer: {answer}
Consider:
1. Mathematical correctness
2. Step-by-step reasoning
3. Final answer accuracy
Provide a score between 0 and 1:"""
        
        score = float(self.coordinator.generate_response(prompt, max_tokens=10))
        return min(1.0, max(0.0, score))
    
    def get_collaborative_reward(self, answer: str, 
                               other_answers: List[str], 
                               commitment: str) -> float:
        """Use LLM to evaluate collaborative contribution."""
        prompt = f"""Evaluate this solution's collaborative value:
Solution: {answer}
Other solutions: {other_answers}
Team commitment: {commitment}
Consider:
1. Unique contribution
2. Complementarity with other solutions
3. Alignment with team commitment
Score (0-1):"""
        
        score = float(self.coordinator.generate_response(prompt, max_tokens=10))
        return min(1.0, max(0.0, score))
    def _calculate_weight_adjustment_loss(self, rewards_history: Dict[str, List[float]]) -> torch.Tensor:
        """
        Calculates the reward difference loss ($L_{dr}$), used for weight adjustment, based on the ECON paper.
        L_dr = Σ(r_i^actual - r_i^expected)²
        
        Args:
            rewards_history (dict): A history dictionary containing records for various reward components, e.g., {'al': [...], 'ts': [...], 'cc': [...]}.
            
        Returns:
            float: The loss value used for weight adjustment.
        """
        
        reward_lengths = [len(rewards) for rewards in rewards_history.values()]
        if len(set(reward_lengths)) != 1:
            logger.warning(f"Inconsistent reward history lengths: {reward_lengths}")
            return torch.tensor(0.0)
            
       
        reward_al = torch.tensor(rewards_history['al'], dtype=torch.float32)
        reward_ts = torch.tensor(rewards_history['ts'], dtype=torch.float32)
        reward_cc = torch.tensor(rewards_history['cc'], dtype=torch.float32)
        
        
        rewards_actual = torch.stack([reward_al, reward_ts, reward_cc], dim=1)  # [batch, 3]
        
        
        rewards_expected = self.weights.expand(rewards_actual.shape[0], -1)  # [batch, 3]
        
        
        reward_diff_loss = torch.sum((rewards_actual - rewards_expected) ** 2)
        
        return reward_diff_loss
    
    def update_weights(self, rewards_history: Dict[str, List[float]]):
        """
        Dynamically adjusts weights based on the reward difference loss.

        Args:
            rewards_history (dict): A history dictionary containing records for various reward components, e.g., {'al': [...], 'ts': [...], 'cc': [...]}.
        """
        
        loss = self._calculate_weight_adjustment_loss(rewards_history)
        
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        with torch.no_grad():
            self.weights.data = F.softmax(self.weights.data, dim=0)
            
        logger.info(f"Updated reward weights: {self.weights.data}")