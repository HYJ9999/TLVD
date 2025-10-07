import copy
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from components.episode_buffer import EpisodeBatch
from modules.mixer.mix_llm import LLMQMixer
from modules.belief_encoder import BeliefEncoder
from typing import Dict, List, Tuple, Optional, Any
import os

"""
Q-Learning algorithm with multi-agent coordination.

Learner for the ECON framework.

Implements Q-learning with:
- Multi-agent coordination
- LLM-based belief networks
- Mixing networks for global Q-values
- Dynamic reward systems
- Two-stage belief coordination
- Bayesian Nash Equilibrium (BNE) updates
"""

class ECONLearner:
    """
    Learner for the ECON framework.
    Handles the optimization of individual BeliefNetworks, the BeliefEncoder,
    and the CentralizedMixingNetwork (LLMQMixer).
    """
    
    def __init__(self, mac: Any, scheme: Dict, logger: Any, args: Any):
        self.args = args
        self.logger = logger
        self.mac = mac
        # Correctly access use_cuda attribute
        use_cuda = hasattr(args, 'system') and hasattr(args.system, 'use_cuda') and args.system.use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        
        self.last_target_update_episode = 0
        self.log_stats_t = -getattr(args, "learner_log_interval", 100) - 1

        # Initialize ECON Network Components
        self.mixer: Optional[LLMQMixer] = None
        self.target_mixer: Optional[LLMQMixer] = None
        self.belief_encoder: Optional[BeliefEncoder] = None
        self.target_belief_encoder: Optional[BeliefEncoder] = None
        self.target_mac = None

        # Parameter Groups
        self.belief_net_params: List = []
        self.encoder_params: List = []
        self.mixer_params: List = []

        # Optimizers
        self.belief_optimizer: Optional[Adam] = None
        self.encoder_optimizer: Optional[Adam] = None
        self.mixer_optimizer: Optional[Adam] = None

        # Loss Weights
        self.gamma = getattr(args, "gamma", 0.99)
        self.lambda_e = getattr(args, "lambda_e", 0.1)
        self.lambda_sd = getattr(args, "lambda_sd", 0.1)
        self.lambda_m = getattr(args, "lambda_m", 0.1)
        self.lambda_belief = getattr(args.loss, "belief_weight", 0.1) if hasattr(args, 'loss') else 0.1
        
        
        self.bne_max_iterations = getattr(args, "bne_max_iterations", 5)
        self.bne_convergence_threshold = getattr(args, "bne_convergence_threshold", 0.01)
        self.stage2_weight = getattr(args, "stage2_weight", 0.3)  # Stage 2 weight in total loss
        
        # Initialize networks and optimizers
        self._initialize_networks_and_optimizers(args)

    def _initialize_networks_and_optimizers(self, args: Any):
        # Initialize Mixer (CentralizedMixingNetwork)
        if getattr(args, "use_mixer", True):
            self.mixer = LLMQMixer(args)
            self.target_mixer = LLMQMixer(args)
            self.mixer_params = list(self.mixer.parameters())
            self.logger.info(f"Mixer initialized with {len(self.mixer_params)} parameters.")
        else:
            self.mixer = None
            self.target_mixer = None
            self.mixer_params = []
            self.logger.info("Mixer is disabled.")

        # Initialize Belief Encoder
        if hasattr(self.mac, 'belief_encoder') and self.mac.belief_encoder is not None:
            self.belief_encoder = self.mac.belief_encoder
            self.logger.info("Using BeliefEncoder from MAC.")
        elif getattr(args, "use_belief_encoder", True):
            self.belief_encoder = BeliefEncoder(args)
            self.logger.info("Initialized standalone BeliefEncoder.")
        else:
            self.belief_encoder = None
            self.logger.info("BeliefEncoder is disabled.")
        
        if self.belief_encoder is not None:
            self.encoder_params = list(self.belief_encoder.parameters())
            self.target_belief_encoder = copy.deepcopy(self.belief_encoder)
            self.logger.info(f"BeliefEncoder has {len(self.encoder_params)} parameters.")
        else:
            self.encoder_params = []
            self.target_belief_encoder = None
            
        # Initialize Target MAC
        self.target_mac = copy.deepcopy(self.mac)

        # Collect parameters for Individual Belief Networks
        self.belief_net_params = []
        if hasattr(self.mac, 'agents') and (isinstance(self.mac.agents, list) or isinstance(self.mac.agents, nn.ModuleList)):
            for agent_module in self.mac.agents:
                if hasattr(agent_module, 'belief_network') and agent_module.belief_network is not None:
                    self.belief_net_params.extend(list(agent_module.belief_network.parameters()))
                else:
                    self.logger.warning("An agent module in mac.agents is missing 'belief_network' or it's None.")
        elif hasattr(self.mac, 'agent') and hasattr(self.mac.agent, 'belief_network') and self.mac.agent.belief_network is not None: 
            self.logger.info("Treating mac.agent as the single BeliefNetwork provider.")
            self.belief_net_params.extend(list(self.mac.agent.belief_network.parameters()))
        else:
            self.logger.error("ECONLearner: Could not find belief_network parameters in MAC structure. BeliefNetwork losses might not work.")

        # Initialize Optimizers
        self.belief_optimizer = None
        if self.belief_net_params:
            self.belief_optimizer = Adam(
                params=filter(lambda p: p.requires_grad, self.belief_net_params),
                lr=getattr(args, "belief_net_lr", args.lr),
                weight_decay=getattr(args, "weight_decay", 0.0)
            )
        
        self.encoder_optimizer = None
        if self.encoder_params and self.belief_encoder:
            self.encoder_optimizer = Adam(
                params=filter(lambda p: p.requires_grad, self.encoder_params),
                lr=getattr(args, "encoder_lr", args.lr),
                weight_decay=getattr(args, "weight_decay", 0.0)
            )
        
        self.mixer_optimizer = None
        if self.mixer_params and self.mixer:
            self.mixer_optimizer = Adam(
                params=filter(lambda p: p.requires_grad, self.mixer_params),
                lr=getattr(args, "mixer_lr", args.lr),
                weight_decay=getattr(args, "weight_decay", 0.0)
            )

        if self.mixer is None:
            self.logger.warning("ECONLearner: Mixer is None. Global Q-value calculation and related losses will be skipped during training.")
        if self.belief_encoder is None:
            self.logger.warning("ECONLearner: BeliefEncoder is None. Group representation E and related losses will be skipped.")
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int) -> Dict:
        """
        Train the ECON framework using the provided batch data with two-stage coordination.
        
        Args:
            batch: Episode batch data
            t_env: Current environment timestep
            episode_num: Current episode number
            
        Returns:
            Dictionary containing training statistics
        """
        rewards = batch["reward"][:, :-1].to(self.device)
        terminated = batch["terminated"][:, :-1].float().to(self.device)
        mask = batch["filled"][:, :-1].float().to(self.device)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        if self.mixer is None:
            self.logger.warning("Mixer is None, training will be skipped.")
            return {"status": "skipped_mixer_none"}

        if hasattr(self.mac, 'init_hidden'):
            self.mac.init_hidden(batch.batch_size)
        if hasattr(self.target_mac, 'init_hidden'):
            self.target_mac.init_hidden(batch.batch_size)

        # ===========================================
        # Stage 1: Individual Belief Formation
        # ===========================================
        
        # Collect data from forward passes - Stage 1
        list_belief_states_stage1, list_prompt_embeddings_stage1, list_local_q_values_stage1, list_group_repr_stage1 = [], [], [], []
        list_belief_states_stage1_next, list_prompt_embeddings_stage1_next, list_local_q_values_stage1_next, list_group_repr_stage1_next = [], [], [], []
        
        # Store commitment features if available in batch
        list_commitment_features_t = [] 
        has_commitment_features_in_batch = "commitment_embedding" in batch.scheme
        
        self.logger.debug(f"Commitment embedding in batch scheme: {has_commitment_features_in_batch}")
        if has_commitment_features_in_batch:
            self.logger.debug(f"Commitment embedding scheme: {batch.scheme['commitment_embedding']}")

        # Stage 1: Forward pass through time steps for individual belief formation
        self.logger.debug("Starting Stage 1: Individual belief formation")
        for t in range(batch.max_seq_length - 1):
            print('====+',t)
            _, mac_info_t = self.mac.forward(batch, t, train_mode=True)
            list_belief_states_stage1.append(mac_info_t["belief_states"])
            list_prompt_embeddings_stage1.append(mac_info_t["prompt_embeddings"])
            list_local_q_values_stage1.append(mac_info_t["q_values"])
            list_group_repr_stage1.append(mac_info_t["group_repr"])

            
            if has_commitment_features_in_batch:
                if t < batch.max_seq_length - 1:  # Ensure time step is valid
                    try:
                        commitment_emb_t = batch["commitment_embedding"][:, t]
                        list_commitment_features_t.append(commitment_emb_t)
                        self.logger.debug(f"Added commitment_embedding at t={t}, shape: {commitment_emb_t.shape}")
                    except (KeyError, IndexError) as e:
                        self.logger.warning(f"Failed to get commitment_embedding at t={t}: {e}")
                        # Create dummy commitment embedding
                        dummy_emb = torch.zeros(batch.batch_size, self.args.commitment_embedding_dim, device=self.device)
                        list_commitment_features_t.append(dummy_emb)
                        self.logger.debug(f"Created dummy commitment_embedding at t={t}")

            _, target_mac_info_t_next = self.target_mac.forward(batch, t + 1, train_mode=True)
            list_belief_states_stage1_next.append(target_mac_info_t_next["belief_states"])
            list_prompt_embeddings_stage1_next.append(target_mac_info_t_next["prompt_embeddings"])
            list_local_q_values_stage1_next.append(target_mac_info_t_next["q_values"])
            list_group_repr_stage1_next.append(target_mac_info_t_next["group_repr"])
            del target_mac_info_t_next

        # Stack temporal data for Stage 1
        belief_states_stage1_stacked = torch.stack(list_belief_states_stage1, dim=1)
        prompt_embeddings_stage1_stacked = torch.stack(list_prompt_embeddings_stage1, dim=1)
        local_q_values_stage1_stacked = torch.stack(list_local_q_values_stage1, dim=1)
        group_representation_stage1_stacked = torch.stack(list_group_repr_stage1, dim=1)

        belief_states_stage1_next_stacked = torch.stack(list_belief_states_stage1_next, dim=1)
        local_q_values_stage1_next_stacked = torch.stack(list_local_q_values_stage1_next, dim=1)

        # ===========================================
        # Stage 2: BNE Coordination 
        # ===========================================
        
        self.logger.debug("Starting Stage 2: BNE coordination")
        belief_states_stage2, prompt_embeddings_stage2, local_q_values_stage2, group_representation_stage2 = self._perform_bne_coordination(
            belief_states_stage1_stacked,
            prompt_embeddings_stage1_stacked,
            local_q_values_stage1_stacked,
            group_representation_stage1_stacked,
            batch
        )

        
        commitment_features_t_stacked = None
        if has_commitment_features_in_batch and list_commitment_features_t:
            try:
                commitment_features_t_stacked = torch.stack(list_commitment_features_t, dim=1)
                self.logger.debug(f"Stacked commitment_features shape: {commitment_features_t_stacked.shape}")
            except Exception as e:
                self.logger.warning(f"Failed to stack commitment_features: {e}")
                
                commitment_features_t_stacked = torch.zeros(
                    batch.batch_size, batch.max_seq_length - 1, self.args.commitment_embedding_dim, 
                    device=self.device
                )
                self.logger.debug(f"Created dummy commitment_features_t_stacked shape: {commitment_features_t_stacked.shape}")
        elif has_commitment_features_in_batch:
            
            commitment_features_t_stacked = torch.zeros(
                batch.batch_size, batch.max_seq_length - 1, self.args.commitment_embedding_dim, 
                device=self.device
            )
            self.logger.debug(f"Created dummy commitment_features (empty list) shape: {commitment_features_t_stacked.shape}")

        bs_x_seq_len = batch.batch_size * (batch.max_seq_length - 1)

        # ===========================================
        # Loss Calculation
        # ===========================================

        
        prompt_embeddings_stage2_flat = prompt_embeddings_stage2.reshape(bs_x_seq_len, self.n_agents, -1)
        local_q_values_stage2_flat = local_q_values_stage2.reshape(bs_x_seq_len, self.n_agents)
        group_representation_stage2_flat = group_representation_stage2.reshape(bs_x_seq_len, -1)

        # Target values using Stage 1 (more stable)
        local_q_values_stage1_next_flat = local_q_values_stage1_next_stacked.reshape(bs_x_seq_len, self.n_agents)

       
        commitment_features_flat = None
        if commitment_features_t_stacked is not None:
            commitment_features_flat = commitment_features_t_stacked.reshape(bs_x_seq_len, -1)
            self.logger.debug(f"Flattened commitment_features shape: {commitment_features_flat.shape}")

        # Forward pass through mixer using Stage 2 coordinated values
        mixer_results_stage2 = self.mixer(
            local_q_values=local_q_values_stage2_flat,
            prompt_embeddings=prompt_embeddings_stage2_flat,
            group_representation=group_representation_stage2_flat
        )
        q_total_stage2_flat = mixer_results_stage2["Q_tot"] 

        # Target mixer forward pass using Stage 1 next values，
        
        target_group_repr_next = self.target_belief_encoder(belief_states_stage1_next_stacked.reshape(bs_x_seq_len, self.n_agents, -1)).reshape(bs_x_seq_len, -1)
        target_prompt_embeddings_next_flat = torch.stack(list_prompt_embeddings_stage1_next, dim=1).reshape(bs_x_seq_len, self.n_agents, -1)
        
        target_mixer_results_next = self.target_mixer(
            local_q_values=local_q_values_stage1_next_flat,
            prompt_embeddings=target_prompt_embeddings_next_flat,
            group_representation=target_group_repr_next
        )
        q_total_target_next_flat = target_mixer_results_next["Q_tot"].detach()

        # Prepare reward and termination data
        # rewards_flat = rewards.reshape(bs_x_seq_len, 1)
        # terminated_flat = terminated.reshape(bs_x_seq_len, 1)
        #for test
        rewards_flat = rewards.reshape(bs_x_seq_len, 1).squeeze(-1)
        terminated_flat = terminated.reshape(bs_x_seq_len, 1).squeeze(-1)
        mask_flat = mask.reshape(bs_x_seq_len, 1)

        # Calculate target Q-values
        target_q_total_flat = rewards_flat + self.gamma * (1 - terminated_flat) * q_total_target_next_flat

        # ===========================================
        # BeliefNetwork Loss Calculation
        # ===========================================
        
        belief_loss = self._calculate_belief_network_loss(
            belief_states_stage1_stacked,
            belief_states_stage2,
            local_q_values_stage1_stacked.squeeze(-1),
            local_q_values_stage2.squeeze(-1),
            target_q_total_flat.reshape(batch.batch_size, batch.max_seq_length - 1),
            rewards.squeeze(-1),
            mask.squeeze(-1)
        )

        # ===========================================
        # Mixer Loss Calculation
        # ===========================================
        
        F_i_for_LSD = mixer_results_stage2.get("F_i_for_LSD")
        
        
        self.logger.debug(f"F_i_for_LSD is None: {F_i_for_LSD is None}")
        self.logger.debug(f"commitment_features_flat is None: {commitment_features_flat is None}")
        self.logger.debug(f"lambda_sd: {self.lambda_sd}")
        
        total_mix_loss, loss_components = self.mixer.calculate_mix_loss(
            Q_tot=q_total_stage2_flat,
            local_q_values=local_q_values_stage2_flat,
            F_i_for_LSD=F_i_for_LSD,
            commitment_text_features=commitment_features_flat,
            target_Q_tot=target_q_total_flat,
            rewards_total=rewards_flat,
            gamma=self.gamma,
            lambda_sd=self.lambda_sd,
            lambda_m=self.lambda_m,
            terminated=terminated_flat,
            mask_flat=mask_flat
        )

        # ===========================================
        # BeliefEncoder Loss
        # ===========================================
        
        encoder_loss = self._calculate_encoder_loss(
            belief_states_stage1_stacked,
            belief_states_stage2,
            group_representation_stage1_stacked,
            group_representation_stage2
        )

        # ===========================================
        # Network Optimization - combine losses to avoid retain_graph
        # ===========================================
        
        
        total_loss = belief_loss + encoder_loss + total_mix_loss
        
       
        if self.belief_optimizer:
            self.belief_optimizer.zero_grad()
        if self.encoder_optimizer:
            self.encoder_optimizer.zero_grad()
        if self.mixer_optimizer:
            self.mixer_optimizer.zero_grad()
        
        
        total_loss.backward()
        
        
        if self.belief_optimizer:
            torch.nn.utils.clip_grad_norm_(self.belief_net_params, 10.0)
            self.belief_optimizer.step()
            
        if self.encoder_optimizer:
            torch.nn.utils.clip_grad_norm_(self.encoder_params, 10.0)
            self.encoder_optimizer.step()
            
        if self.mixer_optimizer:
            torch.nn.utils.clip_grad_norm_(self.mixer_params, 10.0)
            self.mixer_optimizer.step()

        # Update target networks periodically
        if episode_num - self.last_target_update_episode >= getattr(self.args, "target_update_interval", 200):
            self._update_targets()
            self.last_target_update_episode = episode_num

        # Prepare training statistics
        train_stats = {
            "loss_total": (belief_loss + encoder_loss + total_mix_loss).item(),
            "loss_belief": belief_loss.item(),
            "loss_encoder": encoder_loss.item(),
            "loss_mixer": total_mix_loss.item(),
            "q_total_stage1_mean": torch.stack(list_local_q_values_stage1).mean().item(),
            "q_total_stage2_mean": local_q_values_stage2.mean().item(),
            "reward_mean": rewards_flat.mean().item(),
        }
        
        # Add individual loss components
        for key, value in loss_components.items():
            if isinstance(value, torch.Tensor):
                train_stats[f"mixer_{key}"] = value.item()
            else:
                train_stats[f"mixer_{key}"] = value

        return train_stats

    def _perform_bne_coordination(self, belief_states_stage1: torch.Tensor, 
                                 prompt_embeddings_stage1: torch.Tensor,
                                 local_q_values_stage1: torch.Tensor,
                                 group_representation_stage1: torch.Tensor,
                                 batch: EpisodeBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute Bayesian Nash Equilibrium coordination to update Stage 2 beliefs
        
        Args:
            belief_states_stage1: Stage 1 belief states
            prompt_embeddings_stage1: Stage 1 prompt embeddings
            local_q_values_stage1: Stage 1 local Q values
            group_representation_stage1: Stage 1 group representation
            batch: Episode batch data
            
        Returns:
            Tuple of (belief_states_stage2, prompt_embeddings_stage2, local_q_values_stage2, group_representation_stage2)
        """
        batch_size, seq_len, n_agents, belief_dim = belief_states_stage1.shape
        
        
        belief_states_current = belief_states_stage1.detach().clone()
        prompt_embeddings_current = prompt_embeddings_stage1.detach().clone()
        local_q_values_current = local_q_values_stage1.detach().clone()
        group_representation_current = group_representation_stage1.detach().clone()
        
        
        for iteration in range(self.bne_max_iterations):
            belief_states_prev = belief_states_current.clone()
            
            
            new_belief_states = torch.zeros_like(belief_states_current)
            new_prompt_embeddings = torch.zeros_like(prompt_embeddings_current)
            new_local_q_values = torch.zeros_like(local_q_values_current)
            new_group_representation = torch.zeros_like(group_representation_current)
            
            
            for t in range(seq_len):
                
                current_beliefs_t = belief_states_current[:, t]  # (batch, n_agents, belief_dim)
                current_group_repr_t = group_representation_current[:, t]  # (batch, group_dim)
                
                
                agent_interactions = self._calculate_agent_interactions(
                    current_beliefs_t, current_group_repr_t
                )
                
                
                updated_beliefs_t = self._update_beliefs_bne(
                    current_beliefs_t, agent_interactions, batch, t
                )
                
                
                updated_prompt_emb_t, updated_q_vals_t = self._recompute_agent_outputs(
                    updated_beliefs_t, batch, t
                )
                
                
                updated_group_repr_t = self.belief_encoder(updated_beliefs_t)
                
                
                new_belief_states[:, t] = updated_beliefs_t
                new_prompt_embeddings[:, t] = updated_prompt_emb_t
                new_local_q_values[:, t] = updated_q_vals_t
                new_group_representation[:, t] = updated_group_repr_t
            
            
            belief_states_current = new_belief_states
            prompt_embeddings_current = new_prompt_embeddings
            local_q_values_current = new_local_q_values
            group_representation_current = new_group_representation
            
            
            belief_change = torch.norm(belief_states_current - belief_states_prev).item()
            if belief_change < self.bne_convergence_threshold:
                self.logger.debug(f"BNE converged after {iteration + 1} iterations, change: {belief_change:.6f}")
                break
        
        return belief_states_current, prompt_embeddings_current, local_q_values_current, group_representation_current

    def _calculate_agent_interactions(self, beliefs: torch.Tensor, group_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute interaction influence matrix among agents
        
        Args:
            beliefs: (batch, n_agents, belief_dim)
            group_repr: (batch, group_dim)
            
        Returns:
            interaction matrix: (batch, n_agents, n_agents)
        """
        batch_size, n_agents, belief_dim = beliefs.shape
        
        
        beliefs_normalized = F.normalize(beliefs.clone(), p=2, dim=-1)
        similarity_matrix = torch.bmm(beliefs_normalized, beliefs_normalized.transpose(-2, -1))
        
        # Incorporate influence of group representation
        group_influence = group_repr.unsqueeze(1).expand(-1, n_agents, -1)  # (batch, n_agents, group_dim)
        
        # Simplified interaction weight computation
        interaction_weights = torch.softmax(similarity_matrix, dim=-1)
        
        return interaction_weights

    def _update_beliefs_bne(self, beliefs: torch.Tensor, interactions: torch.Tensor, 
                           batch: EpisodeBatch, t: int) -> torch.Tensor:
        """
        Update belief states using BNE mechanism
        
        Args:
            beliefs: (batch, n_agents, belief_dim)
            interactions: (batch, n_agents, n_agents)
            batch: Episode batch
            t: Time step
            
        Returns:
            updated beliefs: (batch, n_agents, belief_dim)
        """
       
        batch_size, n_agents, belief_dim = beliefs.shape
        
        # BNE update: each agent considers others' influence
        updated_beliefs = beliefs.clone()
        
        for i in range(n_agents):
            # Current agent's belief
            current_belief_i = beliefs[:, i]  # (batch, belief_dim)
            
            # Influence of other agents on agent i
            other_agents_influence = torch.zeros_like(current_belief_i)
            for j in range(n_agents):
                if i != j:
                    interaction_weight = interactions[:, i, j].unsqueeze(-1)  # (batch, 1)
                    # Avoid in-place operations
                    influence_contribution = interaction_weight * beliefs[:, j]
                    other_agents_influence = other_agents_influence + influence_contribution
            
            # BNE update rule: current belief + weighted influence from other agents
            bne_update_rate = 0.1  # Learning rate, can be a hyperparameter
            updated_beliefs[:, i] = current_belief_i + bne_update_rate * other_agents_influence
        
        return updated_beliefs

    def _recompute_agent_outputs(self, updated_beliefs: torch.Tensor, 
                               batch: EpisodeBatch, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recompute agent outputs based on updated belief states
        
        Args:
            updated_beliefs: (batch, n_agents, belief_dim)
            batch: Episode batch
            t: Time step
            
        Returns:
            Tuple of (prompt_embeddings, q_values)
        """
        batch_size, n_agents, belief_dim = updated_beliefs.shape
        
        
        obs_tokens = batch["obs"][:, t]  # (batch_size, n_agents, max_token_len)
        inputs = obs_tokens.reshape(batch_size * n_agents, -1)
        
        
        if hasattr(self.mac.agent, 'belief_network'):
            
            mask = torch.zeros(inputs.shape, dtype=torch.bool, device=self.device)
            
           
            belief_outputs = self.mac.agent.belief_network(inputs, mask)
            
            
            prompt_embeddings = belief_outputs['prompt_embedding'].view(batch_size, n_agents, -1)
            # q_values = belief_outputs['q_value'].view(batch_size, n_agents, -1).squeeze(-1) #raw
            q_values = belief_outputs['q_value'].view(batch_size, n_agents, -1)
            
            return prompt_embeddings, q_values
        else:
            
            prompt_embeddings = torch.randn(batch_size, n_agents, 2, device=self.device)
            q_values = torch.mean(updated_beliefs, dim=-1)  # Simplified Q-value computation
            
            return prompt_embeddings, q_values

    def _calculate_belief_network_loss(self, belief_states_stage1: torch.Tensor,
                                     belief_states_stage2: torch.Tensor,
                                     q_values_stage1: torch.Tensor,
                                     q_values_stage2: torch.Tensor,
                                     target_q_total: torch.Tensor,
                                     rewards: torch.Tensor,
                                     mask: torch.Tensor) -> torch.Tensor:
        """
        Compute BeliefNetwork loss
        Includes:
        1. Stage 1 TD loss
        2. Stage 2 BNE consistency loss
        3. Regularization of belief states
        
        Args:
            belief_states_stage1/stage2: (batch, seq_len, n_agents, belief_dim)
            q_values_stage1/stage2: (batch, seq_len, n_agents)
            target_q_total: (batch, seq_len)
            rewards: (batch, seq_len)
            mask: (batch, seq_len)
            
        Returns:
            total belief loss
        """
        batch_size, seq_len, n_agents = q_values_stage1.shape
        
        # 1. Stage 1 TD Loss (individual learning)
        target_q_expanded = target_q_total.unsqueeze(-1).expand(-1, -1, n_agents)
        td_error_stage1 = (q_values_stage1 - target_q_expanded.detach()) * mask.unsqueeze(-1) # How is it calculated specifically
        loss_td_stage1 = (td_error_stage1 ** 2).sum() / mask.sum().clamp(min=1e-6)
        
        # 2. Stage 2 BNE Consistency Loss (coordination consistency)

        q_mean_stage2 = q_values_stage2.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        consistency_error = (q_values_stage2 - q_mean_stage2) * mask.unsqueeze(-1)
        loss_bne_consistency = (consistency_error ** 2).sum() / mask.sum().clamp(min=1e-6)
        
        # 3. Belief Evolution Loss (measure reasonable evolution from Stage 1 to Stage 2)
        belief_evolution = belief_states_stage2 - belief_states_stage1
        evolution_norm = torch.norm(belief_evolution, p=2, dim=-1)  # (batch, seq_len, n_agents)
        
        target_evolution_norm = 0.1  
        evolution_loss = ((evolution_norm - target_evolution_norm) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1e-6)
        
        # 4. Belief Regularization (prevent beliefs from being overly complex)
        belief_reg_stage1 = torch.norm(belief_states_stage1, p=2, dim=-1).mean()
        belief_reg_stage2 = torch.norm(belief_states_stage2, p=2, dim=-1).mean()
        
        # Total BeliefNetwork loss
        total_belief_loss = (
            loss_td_stage1 + 
            self.stage2_weight * loss_bne_consistency + 
            0.1 * evolution_loss + 
            0.01 * (belief_reg_stage1 + belief_reg_stage2)
        )
        
        return total_belief_loss

    def _calculate_encoder_loss(self, belief_states_stage1: torch.Tensor,
                              belief_states_stage2: torch.Tensor,
                              group_repr_stage1: torch.Tensor,
                              group_repr_stage2: torch.Tensor) -> torch.Tensor:
        """
        Compute BeliefEncoder loss
        
        Args:
            belief_states_stage1/stage2: (batch, seq_len, n_agents, belief_dim)
            group_repr_stage1/stage2: (batch, seq_len, group_dim)
            
        Returns:
            encoder loss
        """
        # 1. Representation Consistency Loss
        
        batch_size, seq_len, n_agents, belief_dim = belief_states_stage1.shape
        
        
        beliefs_stage1_flat = belief_states_stage1.reshape(-1, n_agents, belief_dim)
        beliefs_stage2_flat = belief_states_stage2.reshape(-1, n_agents, belief_dim)
        
        recomputed_group_repr_stage1 = self.belief_encoder(beliefs_stage1_flat).reshape(batch_size, seq_len, -1)
        recomputed_group_repr_stage2 = self.belief_encoder(beliefs_stage2_flat).reshape(batch_size, seq_len, -1)
        
        
        consistency_loss_stage1 = F.mse_loss(recomputed_group_repr_stage1, group_repr_stage1)
        consistency_loss_stage2 = F.mse_loss(recomputed_group_repr_stage2, group_repr_stage2)
        
        # 2. Evolution Smoothness Loss
        
        evolution_loss = F.mse_loss(group_repr_stage2, group_repr_stage1)
        
        # 3. Representation Diversity Loss
        
        group_repr_stage2_norm = F.normalize(group_repr_stage2.reshape(-1, group_repr_stage2.shape[-1]), p=2, dim=-1)
        diversity_matrix = torch.mm(group_repr_stage2_norm, group_repr_stage2_norm.t())
        diversity_loss = torch.mean(torch.abs(diversity_matrix - torch.eye(diversity_matrix.shape[0], device=self.device)))
        
        total_encoder_loss = (
            consistency_loss_stage1 + consistency_loss_stage2 + 
            0.1 * evolution_loss + 
            0.01 * diversity_loss
        )
        
        return total_encoder_loss

    def _update_targets(self):
        """Update target networks with current network parameters."""
        if self.target_mixer and self.mixer:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.target_belief_encoder and self.belief_encoder:
            self.target_belief_encoder.load_state_dict(self.belief_encoder.state_dict())
        if self.target_mac and self.mac:
            self.target_mac.load_state_dict(self.mac.state_dict())

    def cuda(self):
        """Move all components to CUDA."""
        self.mac.cuda()
        if self.target_mac:
            self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
        if self.target_mixer is not None:
            self.target_mixer.cuda()
        if self.belief_encoder is not None: 
            self.belief_encoder.cuda()
        if self.target_belief_encoder is not None: 
            self.target_belief_encoder.cuda()

    def save_models(self, path: str):
        """Save all model components."""
        self.mac.save_models(path)
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), f"{path}/mixer.th")
        if self.belief_encoder is not None and not hasattr(self.mac, 'belief_encoder'):
             torch.save(self.belief_encoder.state_dict(), f"{path}/belief_encoder.th")
        
        # Save optimizers for checkpointing
        if self.belief_optimizer:
            torch.save(self.belief_optimizer.state_dict(), f"{path}/belief_opt.pth")
        if self.encoder_optimizer:
            torch.save(self.encoder_optimizer.state_dict(), f"{path}/encoder_opt.pth")
        if self.mixer_optimizer:
            torch.save(self.mixer_optimizer.state_dict(), f"{path}/mixer_opt.pth")

    def load_models(self, path: str):
        """Load all model components."""
        self.mac.load_models(path)
        if self.mixer is not None and os.path.exists(f"{path}/mixer.th"):
            self.mixer.load_state_dict(torch.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage))
        
        if self.belief_encoder is not None and not hasattr(self.mac, 'belief_encoder') and os.path.exists(f"{path}/belief_encoder.th"):
            self.belief_encoder.load_state_dict(torch.load(f"{path}/belief_encoder.th", map_location=lambda storage, loc: storage))

        self._update_targets()

        # Load optimizers if they exist
        if self.belief_optimizer and os.path.exists(f"{path}/belief_opt.pth"):
            self.belief_optimizer.load_state_dict(torch.load(f"{path}/belief_opt.pth"))
        if self.encoder_optimizer and os.path.exists(f"{path}/encoder_opt.pth"):
            self.encoder_optimizer.load_state_dict(torch.load(f"{path}/encoder_opt.pth"))
        if self.mixer_optimizer and os.path.exists(f"{path}/mixer_opt.pth"):
            self.mixer_optimizer.load_state_dict(torch.load(f"{path}/mixer_opt.pth"))