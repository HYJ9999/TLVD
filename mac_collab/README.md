# Causal Evidence Search Reinforcement Learning Implementation Guide

## 1. Overall Architecture

### 1.1 Core Components
- CausalEvidenceEnv: Environment for causal evidence search
- CausalEvidenceLearner: Learner based on the ECON framework
- BeliefEncoder: Belief encoder
- Training script (train.py): Implements the two-stage training workflow

### 1.2 Main Features
- Latent variable description generation and verification
- Multi-source evidence search (papers, Wikipedia)
- Belief state updating and encoding
- BNE coordination mechanism

## 2. Key Implementation Details

### 2.1 Loss Function Computation
- Total loss = belief loss + encoder loss + mixer loss
- Keep consistent with the original framework's Q-learning loss function
- In Stage 1, ensure belief_states_stage1 and belief_states_stage2, q_values, and group_representation remain aligned

### 2.2 Latent Variable Description Generation
- Use LLM to generate semantic descriptions
- Infer based on observed variables and known descriptions
- Use system prompts to ensure consistency and uniqueness
- Verify descriptions via multi-source evidence

### 2.3 Edge Description Processing
- Use replace_all_desc to substitute edge descriptions
- Split by || into a list of keywords
- Process keywords (remove trailing periods)
- Handle and propagate failure states

### 2.4 Reward Mechanism
```python
REWARD_CONFIG = {
    'paper_found_reward': 1.0,      # Reward for finding a paper
    'paper_evidence_reward': 2.0,   # Reward for paper evidence strength
    'wiki_found_reward': 1.0,       # Reward for finding a Wikipedia entry
    'wiki_evidence_reward': 2.0,    # Reward for Wikipedia evidence strength
    'keyword_change_penalty': 0.1,  # Penalty for changing keywords
    'description_failure_penalty': -1.0,  # Penalty for description generation failure
    'verification_success_reward': 5.0    # Reward for successful verification
}
```

## 3. Training Workflow

### 3.1 Belief Pretraining Phase
- Collect trajectories in forced-exploration mode
- Update the belief encoder
- Initialize belief states

### 3.2 Main Training Phase
- Trajectory collection and updates
- Periodic BNE coordination
- Verification state tracking
- Logging and model saving

## 4. State Tracking

### 4.1 Verification State
- verified_correct: Set of successfully verified latent variables
- failed_or_unverified: Set of failed or unverified latent variables
- history: Error history records

### 4.2 Agent State
- belief_state: Current belief state
- group_representation: Group representation
- current_keywords: Current search keywords
- evidence_strength: Evidence strength

## 5. Key Improvements

### 5.1 Description Generation Improvements
- Integrate description generation logic from agents_collab.py
- Add semantic consistency checks
- Improve error handling mechanisms

### 5.2 Evidence Search Improvements
- Use processed keywords for search
- Cross-validate multi-source evidence
- Dynamically update evidence strength

### 5.3 Training Process Improvements
- Two-stage training strategy
- BNE coordination mechanism
- Dynamic reward adjustment

## 6. Configuration Guide

### 6.1 Environment Configuration
```python
ENV_CONFIG = {
    'graph_path': 'path to graph file',
    'evidence_threshold': 0.7,  # Verification success threshold
    'max_steps': 100,           # Max steps per episode
    'belief_dim': 64,           # Belief state dimension
}
```

### 6.2 Training Configuration
- batch_size: Training batch size
- learning_rate: Learning rate
- gamma: Discount factor
- target_update_interval: Target network update interval
- bne_coordination_interval: BNE coordination interval

## 7. Usage

### 7.1 Training Command
```bash
python train.py --config path/to/config.yaml
```

### 7.2 Configuration File Example
```yaml
system:
  use_cuda: true
  seed: 42

env:
  graph_path: "path/to/graph.dot"
  evidence_threshold: 0.7

training:
  batch_size: 32
  learning_rate: 0.001
  gamma: 0.99
  episodes: 1000
  bne_interval: 100
```

## 8. Notes
1. Ensure the correct graph file path is provided
2. Check the format of the variable description files
3. Monitor verification status and error history
4. Save models and training logs regularly
5. Adjust reward parameters when appropriate