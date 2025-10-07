from loguru import logger
import os
from datetime import datetime

class SimpleLogger:
    """Simple Logger"""
    def __init__(self, args):
        self.args = args
        
        
        if hasattr(args, 'log_dir'):
            log_dir = args.log_dir
        else:
            log_dir = "logs"
            
        
        os.makedirs(log_dir, exist_ok=True)
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"train_{timestamp}.log")
        
        
        logger.add(
            log_file,
            rotation="500 MB",  
            retention="10 days",  
            level="INFO"
        )
        
        self.logger = logger
        
    def info(self, msg):
        """Log information-level messages"""
        self.logger.info(msg)
        
    def warning(self, msg):
        """Log warning-level messages"""
        self.logger.warning(msg)
        
    def error(self, msg):
        """Log error-level messages"""
        self.logger.error(msg)
        
    def debug(self, msg):
        """Log debug-level messages"""
        self.logger.debug(msg)
        
    def critical(self, msg):
        """Log critical-level messages"""
        self.logger.critical(msg)
        
    def log_training_info(self, epoch, train_info):
        """Log train messages"""
        self.info(f"Epoch {epoch}:")
        for key, value in train_info.items():
            self.info(f"  {key}: {value}")
            
    def log_evaluation_info(self, eval_info):
        """Log evaluation messages"""
        self.info("Evaluation Results:")
        for key, value in eval_info.items():
            self.info(f"  {key}: {value}")
            
    def log_belief_update(self, agent_type, old_belief, new_belief, evidence_strength):
        """Log belief update messages"""
        self.debug(f"{agent_type} Belief Update:")
        self.debug(f"  Evidence Strength: {evidence_strength}")
        self.debug(f"  Old Belief: {old_belief}")
        self.debug(f"  New Belief: {new_belief}")
        
    def log_action_info(self, agent_type, action, reward):
        """Log action messages"""
        self.debug(f"{agent_type} Action Info:")
        self.debug(f"  Selected Action: {action}")
        self.debug(f"  Reward Obtained: {reward}")
        
    def log_evidence_found(self, agent_type, evidence_type, strength):
        """Log evidence messages"""
        self.info(f"{agent_type} Found New Evidence:")
        self.info(f"  Evidence Type: {evidence_type}")
        self.info(f"  Evidence Strength: {strength}")