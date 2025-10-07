#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any
import wandb

# Import necessary components
from utils.logging import get_logger
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY

from paper_search import token_counter
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ECON Framework Training Script')
    
    # Basic parameters
    parser.add_argument('--config', type=str, default='/home/swufe/DHM/ECON/src/config/config.yaml', help='Configuration file path')
    parser.add_argument('--executor_model', type=str, help='Executor LLM model name')
    parser.add_argument('--coordinator_model', type=str, help='Coordinator LLM model name')
    parser.add_argument('--n_agents', type=int, default=2,help='Number of agents')
    parser.add_argument('--experiment_name', type=str,default="econ-gsm8k-training", help='Experiment name')
    parser.add_argument('--log_dir', type=str, help='Log directory')
    parser.add_argument('--api_key', default='540b8796f5b36eef674ba2b7d7246cd2b08e1ef105f49e5b053f002b38644d16', help='Together API key')#'sk-zQDMqI1Mi4AGnqHQ9cDb0b28Ce4b4891A8AdEe48007e1bDe'
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--env', type=str, help='Environment name')
    
    # wandb related parameters
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb for experiment logging')
    parser.add_argument('--wandb_project', type=str, default='ECON-Framework', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, help='wandb username or organization name')
    parser.add_argument('--wandb_tags', type=str, help='wandb tags, separated by commas')
    
    return parser.parse_args()

def load_config(config_path: str) -> SimpleNamespace:
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Recursively convert to SimpleNamespace for easy access
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d
    
    config = dict_to_namespace(config_dict)
    return config

def update_config_with_args(config: SimpleNamespace, args: Any) -> SimpleNamespace:
    """Update configuration with command line arguments"""
    # Only update non-None parameters
    if args.executor_model:
        if hasattr(config, 'llm'):
            config.llm.executor_model = args.executor_model
    
    if args.coordinator_model:
        if hasattr(config, 'llm'):
            config.llm.coordinator_model = args.coordinator_model
    
    if args.n_agents:
        config.n_agents = args.n_agents
    
    if args.experiment_name:
        if hasattr(config, 'logging'):
            config.logging.experiment_name = args.experiment_name
    
    if args.log_dir:
        if hasattr(config, 'logging'):
            config.logging.log_path = args.log_dir
    
    if args.api_key:
        # Set API key in both places for compatibility
        config.together_api_key = args.api_key  # For direct access as args.together_api_key
        if hasattr(config, 'llm'):
            config.llm.together_api_key = args.api_key
    
    if args.seed:
        config.system.seed = args.seed
    
    if args.env:
        config.env = args.env
    
    # Add wandb related configuration
    if not hasattr(config, 'wandb'):
        config.wandb = SimpleNamespace()
    
    config.wandb.use_wandb = args.use_wandb
    if args.wandb_project:
        config.wandb.project = args.wandb_project
    if args.wandb_entity:
        config.wandb.entity = args.wandb_entity
    if args.wandb_tags:
        config.wandb.tags = args.wandb_tags.split(',')
    
    return config

def setup_experiment(config: SimpleNamespace):
    """Setup experiment environment and components"""
    logger = get_logger()
    logger.info("Setting up experiment environment...")
    
    # Set random seed
    seed = config.system.seed if hasattr(config, 'system') and hasattr(config.system, 'seed') else 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set device
    use_cuda = hasattr(config, 'system') and hasattr(config.system, 'use_cuda') and config.system.use_cuda and torch.cuda.is_available()
    device_num = config.system.device_num if hasattr(config, 'system') and hasattr(config.system, 'device_num') else 0
    device = torch.device(f"cuda:{device_num}" if use_cuda else "cpu")
    
    # Add device to config object for runner access
    config.device = device
    
    if use_cuda:
        torch.cuda.set_device(device_num)
    
    # Initialize Runner
    runner = r_REGISTRY[config.runner](args=config, logger=logger)
    
    # Setup schemes and groups
    scheme = {
        "state": {"vshape": runner.env_info["state_shape"]},
        "obs": {
            "vshape": runner.env_info["obs_shape"],
            "group": "agents"
        },
        "actions": {
            "vshape": (1,),
            "group": "agents",
            "dtype": torch.long
        },
        "avail_actions": {
            "vshape": (runner.env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.int
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "llm_responses": {
            "vshape": (config.text_embed_dim,),
            "group": "agents"
        },
        "strategy": {"vshape": (config.text_embed_dim,)},
        "commitment": {"vshape": (config.text_embed_dim,)},
        "belief_states": {
            "vshape": (config.belief_dim,),
            "group": "agents"
        }
    }
    
    groups = {"agents": config.n_agents}
    
    # Initialize MAC
    mac = mac_REGISTRY[config.mac](scheme, groups, config)
    
    # Setup runner
    runner.setup(scheme, groups, None, mac)
    
    # Initialize learner
    learner = le_REGISTRY[config.learner](mac=mac, scheme=scheme, logger=logger, args=config)
    
    return runner, mac, learner, logger, device

def run_training(config: SimpleNamespace, runner, learner, logger, device):
    """Execute training loop"""
    logger.info("Starting training...")
    
    begin_time = time.time()
    
    # Training configuration
    t_max = getattr(config, 't_max', 5)
    test_interval = getattr(config, 'test_interval', 50)
    log_interval = getattr(config, 'logging', SimpleNamespace()).log_interval if hasattr(config, 'logging') else 2000
    save_model_interval = getattr(config, 'logging', SimpleNamespace()).save_model_interval if hasattr(config, 'logging') else 50
    
    # Early stopping related variables
    last_commitment = None
    last_total_loss = None
    patience_counter = 0
    
    # Get early stopping thresholds from configuration
    early_stopping = getattr(config, 'early_stopping', SimpleNamespace())
    commitment_threshold = getattr(early_stopping, 'commitment_threshold', 0.01)
    reward_threshold = getattr(early_stopping, 'reward_threshold', 0.7)
    loss_threshold = getattr(early_stopping, 'loss_threshold', 0.0001)
    patience = getattr(early_stopping, 'patience', 5)

    # Training loop
    episode = 0
    t_env = 0
    
    while t_env < t_max:
        # Run episode
        episode_batch = runner.run(test_mode=False)

        # Train learner
        if episode_batch is not None:
            train_stats = learner.train(episode_batch, t_env, episode)
            
            # Log training statistics
            if episode % log_interval == 0:
                logger.info(f"Episode {episode}, t_env: {t_env}")
                for key, value in train_stats.items():
                    logger.info(f"  {key}: {value}")
                
                # Log to wandb if enabled
                if hasattr(config, 'wandb') and config.wandb.use_wandb:
                    log_to_wandb(train_stats, episode, 'train/')
            
            # Save model periodically
            if episode % save_model_interval == 0 and episode > 0:
                save_path = Path(config.logging.checkpoint_path) / f"episode_{episode}"
                save_path.mkdir(parents=True, exist_ok=True)
                learner.save_models(str(save_path))
                logger.info(f"Model saved at episode {episode}")
        
        episode += 1
        t_env += getattr(config, 'episode_length', 1)
    
    # Final save
    save_path = Path(config.logging.checkpoint_path) / "final"
    save_path.mkdir(parents=True, exist_ok=True)
    learner.save_models(str(save_path))
    logger.info("Training completed. Final model saved.")
    
    total_time = time.time() - begin_time
    logger.info(f"Total training time: {total_time:.2f} seconds")

def run_test(runner, logger):
    """Run test episodes"""
    logger.info("Running test episodes...")
    
    test_episodes = 10
    test_stats = {"test_return": [], "test_success": []}
    
    for _ in range(test_episodes):
        episode_batch = runner.run(test_mode=True)
        if episode_batch is not None:
            # Calculate episode statistics
            episode_return = episode_batch["reward"].sum().item()
            episode_success = 1 if episode_return > 0 else 0
            
            test_stats["test_return"].append(episode_return)
            test_stats["test_success"].append(episode_success)
    
    # Calculate averages
    avg_return = np.mean(test_stats["test_return"]) if test_stats["test_return"] else 0
    success_rate = np.mean(test_stats["test_success"]) if test_stats["test_success"] else 0
    
    return {
        "test_return_mean": avg_return,
        "test_success_rate": success_rate,
        "test_episodes": len(test_stats["test_return"])
    }

def setup_wandb(config: SimpleNamespace, logger):
    """Initialize wandb for experiment tracking"""
    if hasattr(config, 'wandb') and config.wandb.use_wandb:
        logger.info("Initializing wandb...")
        
        wandb.init(
            project=getattr(config.wandb, 'project', 'ECON-Framework'),
            entity=getattr(config.wandb, 'entity', None),
            tags=getattr(config.wandb, 'tags', None),
            config=dict(config.__dict__) if hasattr(config, '__dict__') else None,
            name=getattr(config.logging, 'experiment_name', 'econ_experiment')
        )
        
        logger.info("wandb initialized successfully")
        return True
    return False

def log_to_wandb(data: Dict, step: int, prefix: str = ''):
    """Log data to wandb"""
    if wandb.run is not None:
        wandb.log({f"{prefix}{k}": v for k, v in data.items()}, step=step)

def main():
    """Main training function"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Load configuration
        config = load_config(args.config)
        
        # Update configuration with command line arguments
        config = update_config_with_args(config, args)
        
        # Setup experiment
        runner, mac, learner, logger, device = setup_experiment(config)
        
        # Setup wandb if enabled
        setup_wandb(config, logger)
        
        # Run training
        run_training(config, runner, learner, logger, device)
        
        logger.info("Training completed successfully!")

        # 
        token_counter.write_total_to_results() 

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up wandb
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()
    
