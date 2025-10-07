"""Configuration file"""

# System configuration
SYSTEM_CONFIG = {
    "use_cuda": True,  # Whether to use CUDA
    "device": "cuda:0" if True else "cpu"  # Device
}

# Sampling configuration
SAMPLING_CONFIG = {
    "temperature_min": 0.1,  # Minimum temperature
    "temperature_max": 2.0,  # Maximum temperature
    "p_min": 0.1,  # Minimum repetition penalty
    "p_max": 0.9,  # Maximum repetition penalty
    "max_tokens": 400  # Maximum number of generated tokens
}

# Architecture configuration
ARCH_CONFIG = {
    "entity_dim": 256,  # Entity dimension
    "attention_heads": 4,  # Number of attention heads
    "transformer_blocks": 2,  # Number of Transformer blocks
    "dropout_rate": 0.1,  # Dropout rate
    "vocab_size": 50257  # Vocabulary size
}

# Environment configuration
ENV_CONFIG = {
    "max_steps": 10,  # Maximum steps per episode
    "literature_source": "arxiv",  # Literature source
    "graph_path": "data/huaxi33_alpha0.003_rtscale1_N20.dot",  # Graph file path
    "var_desc_path": "data/var_descriptions.json",  # Variable description file
    "latents_desc_path": "data/hidden_var_descriptions.json"  # Hidden variable description file
}

# Agent configuration
AGENT_CONFIG = {
    "state_dim": 2,  # State dimension
    "action_dim": 2,  # Action dimension (0: keep, 1: change)
    "belief_dim": 64,  # Belief state dimension
    
    # Paper agent parameters
    "paper": {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 0.9,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.995,
        "buffer_size": 1000,
        "batch_size": 32
    },
    
    # Wikipedia agent parameters
    "wiki": {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 0.9,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.995,
        "buffer_size": 1000,
        "batch_size": 32
    }
}

# Network configuration
NETWORK_CONFIG = {
    # Belief encoder parameters
    "belief_encoder": {
        "hidden_dim": 128,  # Hidden layer dimension
        "n_heads": 4,  # Number of attention heads
        "dropout": 0.1,  # Dropout rate
        "learning_rate": 0.0005  # Learning rate
    },
    
    # Mixer parameters
    "mixer": {
        "mixing_embed_dim": 32,  # Mixing embedding dimension
        "hypernet_embed": 64,  # Hypernetwork embedding dimension
        "n_quantiles": 8,  # Number of quantiles
        "learning_rate": 0.001  # Learning rate
    },
    
    # LLM parameters
    "llm": {
        "prompt_dim": 2,  # Prompt embedding dimension
        "max_prompt_length": 128,  # Maximum prompt length
        "temperature": 0.7  # Sampling temperature
    }
}

# Training configuration
TRAIN_CONFIG = {
    "n_episodes": 1000,  # Number of training episodes
    "log_interval": 100,  # Logging interval
    "save_interval": 100,  # Model saving interval
    "target_update_interval": 10,  # Target network update interval
    "seed": 42,  # Random seed
    "log_dir": "logs",  # Log directory
    
    # Two-stage training parameters
    "belief_pretrain_episodes": 100,  # Belief pretraining episodes
    "belief_loss_weight": 0.1,  # Belief loss weight
    "grad_norm_clip": 10.0,  # Gradient clipping threshold
    
    # BNE coordination parameters
    "bne_start_episode": 200,  # Episode to start BNE coordination
    "bne_update_interval": 50,  # BNE update interval
    "bne_temperature": 0.1  # BNE temperature parameter
}

# Reward configuration
REWARD_CONFIG = {
    "paper_found_reward": 1.0,  # Reward for finding a paper
    "paper_evidence_reward": 2.0,  # Coefficient for paper evidence strength
    "wiki_found_reward": 1.0,  # Reward for finding a Wikipedia entry
    "wiki_evidence_reward": 2.0,  # Coefficient for Wikipedia evidence strength
    "keyword_change_penalty": 0.1,  # Penalty for changing keywords
    "description_failure_penalty": -1.0,  # Penalty for failing to generate descriptions
    "verification_success_reward": 5.0  # Reward for successful verification
}