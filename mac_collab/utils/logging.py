import logging
import numpy as np
import torch
import collections
from typing import Dict, Any, Optional, Union, List
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
import os
from pathlib import Path
import sys

class Logger:
    """
    Comprehensive logging utility for LLM-based MARL training.
    
    Handles console logging, TensorBoard logging, and metric tracking
    with support for various data types and formats.
    """
    
    def __init__(self, console_logger: logging.Logger, directory: str, 
                 experiment_name: str, use_tensorboard: bool = True):
        """
        Initialize the logger.
        
        Args:
            console_logger: Base console logger
            directory: Directory for log files
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard
        """
        self.console_logger = console_logger
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        
        # Setup log directory
        self.log_dir = Path(directory) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize stat tracking
        self.stats: Dict[str, collections.deque] = {}
        self.stat_windows: Dict[str, int] = {}
        
        # Setup TensorBoard
        if use_tensorboard:
            self.tb_writer = SummaryWriter(str(self.log_dir / "tb_logs"))
        
        # Setup metric logging
        self.metric_log_path = self.log_dir / "metrics.jsonl"
        
    def info(self, message: str) -> None:
        """Log info message."""
        self.console_logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.console_logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.console_logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.console_logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.console_logger.critical(message)
        
    def setup_custom_logger(name: str, log_dir: str, 
                          level: int = logging.INFO) -> logging.Logger:
        """
        Setup a custom console logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            
        Returns:
            Configured logger instance
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
        )

        handler = logging.FileHandler(
            log_dir / f"{name}.log",
            mode='a'
        )
        handler.setFormatter(formatter)

        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        logger.addHandler(screen_handler)
        
        return logger
        
    def log_stat(self, key: str, value: Union[float, torch.Tensor, np.ndarray], 
                 t: int, window: int = 100) -> None:
        """
        Log a statistic with sliding window averaging.
        
        Args:
            key: Metric key
            value: Metric value
            t: Timestep
            window: Window size for averaging
        """
        if key not in self.stats:
            self.stats[key] = collections.deque(maxlen=window)
            self.stat_windows[key] = window

        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.item()

        self.stats[key].append(value)

        if self.use_tensorboard:
            self.tb_writer.add_scalar(key, value, t)
            
        # Log to metrics file
        with open(self.metric_log_path, 'a') as f:
            log_entry = {
                'time': datetime.now().isoformat(),
                'step': t,
                'metric': key,
                'value': value,
                'window_size': window
            }
            f.write(json.dumps(log_entry) + '\n')

    def print_recent_stats(self) -> None:
        """Print recent statistics to console."""
        log_str = "Recent Stats | "
        
        for k, v in sorted(self.stats.items()):
            if len(v) > 0:
                log_str += f"{k}: {np.mean(list(v)):.3f} | "
                
        self.console_logger.info(log_str)

    def log_metrics(self, metrics: Dict[str, Any], step: int, 
                   prefix: str = '') -> None:
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step
            prefix: Optional prefix for metric names
        """
        for name, value in metrics.items():
            if prefix:
                name = f"{prefix}/{name}"
            self.log_stat(name, value, step)

    def log_episode(self, episode_metrics: Dict[str, Any], episode_num: int,
                   is_training: bool = True) -> None:
        """
        Log episode-specific metrics.
        
        Args:
            episode_metrics: Dictionary of episode metrics
            episode_num: Episode number
            is_training: Whether this is a training episode
        """
        prefix = 'train' if is_training else 'test'
        
        # Log basic episode stats
        self.log_metrics(
            {
                'episode_length': episode_metrics.get('length', 0),
                'episode_return': episode_metrics.get('return', 0),
                'episode_success': int(episode_metrics.get('success', False))
            },
            episode_num,
            prefix=prefix
        )
        
        # Log LLM-specific metrics
        if 'llm_metrics' in episode_metrics:
            self.log_metrics(
                episode_metrics['llm_metrics'],
                episode_num,
                prefix=f"{prefix}/llm"
            )

    def log_model_summary(self, model: torch.nn.Module, name: str) -> None:
        """
        Log model architecture summary.
        
        Args:
            model: PyTorch model
            name: Model name
        """
        if self.use_tensorboard:
            # Log model graph
            dummy_input = model.get_dummy_input()
            self.tb_writer.add_graph(model, dummy_input)
            
            # Log model parameters
            for name, param in model.named_parameters():
                self.tb_writer.add_histogram(f"params/{name}", param, 0)

    def log_grad_norms(self, model: torch.nn.Module, step: int) -> None:
        """
        Log gradient norms for model parameters.
        
        Args:
            model: PyTorch model
            step: Current step
        """
        if not self.use_tensorboard:
            return
            
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.tb_writer.add_histogram(
                    f"grads/{name}",
                    param.grad,
                    step
                )

    def log_belief_states(self, belief_states: torch.Tensor, 
                         step: int, agent_ids: Optional[List[int]] = None) -> None:
        """
        Log belief state visualizations.
        
        Args:
            belief_states: Belief state tensor
            step: Current step
            agent_ids: Optional list of agent IDs
        """
        if not self.use_tensorboard or belief_states is None:
            return
            
        if agent_ids is None:
            agent_ids = list(range(belief_states.shape[0]))
            
        for idx, agent_id in enumerate(agent_ids):
            self.tb_writer.add_histogram(
                f"belief_states/agent_{agent_id}",
                belief_states[idx],
                step
            )

    def log_llm_outputs(self, outputs: Dict[str, str], step: int) -> None:
        """
        Log LLM outputs and analyzed metrics.
        
        Args:
            outputs: Dictionary of LLM outputs
            step: Current step
        """
        # Log to file
        output_log_path = self.log_dir / "llm_outputs.jsonl"
        with open(output_log_path, 'a') as f:
            log_entry = {
                'time': datetime.now().isoformat(),
                'step': step,
                'outputs': outputs
            }
            f.write(json.dumps(log_entry) + '\n')
        
        # Log any numerical metrics to TensorBoard
        if self.use_tensorboard:
            for key, value in outputs.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"llm/{key}", value, step)

    def close(self) -> None:
        """Clean up resources."""
        if self.use_tensorboard:
            self.tb_writer.close()


def get_logger(log_dir: str = "logs", experiment_name: str = None,
              use_tensorboard: bool = True) -> Logger:
    """
    Get a configured logger instance.
    
    Args:
        log_dir: Directory for logs
        experiment_name: Name of the experiment
        use_tensorboard: Whether to use TensorBoard
        
    Returns:
        Configured logger instance
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    console_logger = Logger.setup_custom_logger(
        "console_logger",
        log_dir
    )
    
    return Logger(
        console_logger=console_logger,
        directory=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=use_tensorboard
    )