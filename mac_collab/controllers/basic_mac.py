from re import L
import torch
import torch.nn as nn
from modules.agents.transformer_agent import LLMTransformerAgent
from modules.llm.llm_wrapper import ImprovedLLMWrapper, LLMConfig
from modules.llm.commitment_embedder import CommitmentEmbedder
from components.action_selectors import REGISTRY as action_REGISTRY
from torch.nn import functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from modules.belief_encoder import BeliefEncoder
from transformers import AutoTokenizer
from utils.causal_evidence_search import search_causal_evidence
from utils.latent_desc_saver import save_latent_description ,append_latent_descriptions_to_txt
import json
class LLMBasicMAC:
    """
    Multi-Agent Controller coordinating Transformer-based LLM agents.
    Handles both coordinator and executor roles in the system.
    """
    def __init__(self, scheme: Dict, groups: Dict, args: Any):
        self.n_agents = args.n_agents
        self.args = args
        # Correctly access use_cuda attribute
        use_cuda = hasattr(args, 'system') and hasattr(args.system, 'use_cuda') and args.system.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        # Initialize tokenizer with better error handling
        model_name = args.llm_model_name if hasattr(args, "llm_model_name") else "gpt2"
        try:
            # Try to load tokenizer with local_files_only first
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            logger.info(f"Successfully loaded tokenizer for {model_name} from local cache")
        except (OSError, ConnectionError, Exception) as e:
            logger.warning(f"Failed to load tokenizer for model '{model_name}' from cache: {e}")
            logger.info("Trying to create a simple tokenizer as fallback...")
            try:
                # Create a basic tokenizer as fallback
                from transformers import GPT2Tokenizer
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
                logger.info("Using cached GPT-2 tokenizer as fallback")
            except Exception as e2:
                logger.error(f"Failed to load any tokenizer: {e2}")
                # Create a minimal tokenizer
                self.tokenizer = self._create_minimal_tokenizer()
                logger.info("Created minimal tokenizer as last resort")
                
        # Ensure pad_token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer missing pad_token, set to eos_token: {self.tokenizer.eos_token}")

        # Get input shape for agents (now based on tokenized obs)
        input_shape = self._get_input_shape(scheme) 
        
        # Initialize agents (executors)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        
        # Action selector
        self.action_selector = action_REGISTRY[args.action_selector](args)
        
        # Common LLM Config for Coordinator and Embedder API access
        common_llm_config = LLMConfig(
            api_key=args.together_api_key,
            model_name=args.coordinator_model,
            debug=getattr(args, "debug", False),
            max_retries=getattr(args, "llm_max_retries", 6),
            max_workers=getattr(args, "llm_max_workers", 5)
        )

        # Initialize coordinator LLM
        self.coordinator = ImprovedLLMWrapper(
            api_key=args.together_api_key,
            model_name=args.coordinator_model,
            belief_dim=args.belief_dim,
            debug=getattr(args, "debug", False)
        )
        
        # Initialize BeliefEncoder
        self.belief_encoder = BeliefEncoder(
            belief_dim=args.belief_dim,
            n_agents=args.n_agents,
            n_heads=args.arch.attention_heads if hasattr(args, 'arch') and hasattr(args.arch, 'attention_heads') else 4,
            key_dim=args.arch.key_dim if hasattr(args, 'arch') and hasattr(args.arch, 'key_dim') else 64,
            device=self.device
        )

        # Initialize Commitment Embedder
        self.commitment_embedder = CommitmentEmbedder(args, common_llm_config)
        
        # Response caches with size limits
        self.max_cache_size = getattr(args, 'max_cache_size', 1000)
        self.strategy_cache = {}
        self.commitment_cache = {}
        
        # Initialize attention masks
        self.setup_attention_masks()

     
    def preprocess_observation(self, observation_text: str, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Tokenize and preprocess a single observation text string.
        Args:
            observation_text: The raw text of the observation (e.g., a question).
            max_length: Optional maximum length for padding/truncation. If None, uses args.max_question_length.
        Returns:
            A tensor of token IDs, padded/truncated to max_length.
        """
        if max_length is None:
            max_length = getattr(self.args.env_args, "max_question_length", 512)

        # Tokenize the text
        encoding = self.tokenizer(
            observation_text,
            # add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            # return_attention_mask=False,
            return_tensors='pt',
        )
        
        # .input_ids is typically shape (1, seq_len). We want (seq_len,)
        return encoding['input_ids'].squeeze(0).to(self.device)

    def setup_attention_masks(self):
        """Setup reusable attention masks."""
        self.base_attention_mask = torch.zeros(
            (1, self.args.max_seq_length),
            dtype=torch.bool,
            device=self.device
        )
        
        # Create causal attention mask if needed
        if self.args.use_causal_mask:
            mask = torch.triu(
                torch.ones(self.args.max_seq_length, self.args.max_seq_length),
                diagonal=1
            )
            self.causal_mask = mask.bool().to(self.device)

    def _build_agents(self, input_shape: int):
        """Initialize Transformer agents."""
        try:
            self.agent = LLMTransformerAgent(input_shape, self.args)
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise

    def init_hidden(self, batch_size: int):
        """
        Initialize hidden states (dummy method for interface compatibility).
        For transformers, we initialize positional embeddings instead.
        """
        if not hasattr(self, 'pos_embeddings'):
            self.pos_embeddings = torch.arange(
                0, self.args.max_seq_length,
                device=self.device
            ).unsqueeze(0).expand(batch_size, -1)

    def _build_inputs(self, batch: Any, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct inputs for transformer agents.
        
        Args:
            batch: Batch of episode data
            t: Current timestep
            
        Returns:
            Tuple containing:
                - Input tensor
                - Attention mask
        """
        bs = batch.batch_size
        
        # For token-based input, use observation sequence directly
        # batch["obs"] shape: (batch_size, max_seq_len, n_agents, max_token_len)
        # We need: (batch_size * n_agents, max_token_len)
        obs_tokens = batch["obs"][:, t]  # (batch_size, n_agents, max_token_len)
        inputs = obs_tokens.reshape(bs * self.n_agents, -1).clone()  # (batch_size * n_agents, max_token_len) - use clone to avoid shared memory
        
        # Create attention mask based on token validity
        seq_len = inputs.size(1)  # max_token_len
        
        # Simple padding detection: find pad token positions
        if hasattr(self, 'tokenizer'):
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                pad_token_id = self.tokenizer.pad_token_id
            else:
                # If pad_token_id is None, use eos_token_id
                pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = 50256  # GPT2's eos_token_id
            
        mask = (inputs == pad_token_id)
        
        return inputs, mask

    def select_actions(self, ep_batch: Any, t_ep: int, t_env: int, 
                      raw_observation_text: Optional[str] = None,
                      bs: slice = slice(None),env_step_info: Optional[Dict] = None, 
                      test_mode: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Select actions for all agents and generate LLM responses and commitment features.
        
        Args:
            ep_batch: Episode batch data. NOTE: ep_batch["obs"] is now expected to be tokenized IDs.
            t_ep: Current episode timestep
            t_env: Current environment timestep
            raw_observation_text: Raw observation text for LLM processing
            bs: Batch slice
            test_mode: Whether in test mode
            
        Returns:
            Tuple of (actions, info_dict)
        """
        # Get available actions
        avail_actions = ep_batch["avail_actions"][:, t_ep]# Check how ep_batch["avail_actions"] is obtained specifically
        env_step_info=env_step_info['sample']

        # Forward pass through agents
        agent_outputs, agent_info = self.forward(ep_batch, t_ep, test_mode)
        
        # Select actions based on agent outputs
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        logger.info(f"Chosen actions at t_env {t_env}: {chosen_actions}, agent_outputs[bs]:{ agent_outputs[bs]},avail_actions[bs]:{avail_actions[bs]}")
        # Generate LLM responses if raw text is provided
        if raw_observation_text is not None:
            # Get strategy from coordinator
            # TODO: remove strategy
            strategy = None #self._get_strategy(raw_observation_text)
            
            # Get executor responses - get each agent's generated description
            executor_responses = []
            for agent_id in range(self.n_agents):
                response = self.agent.generate_answer(
                    question=raw_observation_text, 
                    strategy=strategy
                )
                executor_responses.append(response)
            
            logger.info(f"💬 Generated descriptions:")
            for i, response in enumerate(executor_responses):
                logger.info(f"    Agent {i+1}: {response}")

            # TODO: aggregate evidence for each edge
            evidence_strings = []
            evidence_ratio_lsit=[]
            confidence_lsit=[]
            logger.info(f"💬 Casual inference:")
            for agent_id, response in enumerate(executor_responses):
                # Extract evidence from responses
                # Convert response to dictionary
                response_dict = json.loads(response)
                # Keep only L1 description and remove confidence
                latent_desc = {k: v for k, v in response_dict.items() if k != "confidence"}
                append_latent_descriptions_to_txt(latent_desc,agent_id)
                confidence=response_dict["confidence"] if "confidence" in response_dict else 0.0
                confidence_lsit.append(confidence)
                agent_id=agent_id+1
                logger.info(f"Agent {agent_id} latent_desc: {latent_desc}")
                logger.info(f"Agent {agent_id} latent_graph: {env_step_info['latent_graph']}")
                # print("var_names_and_desc",env_step_info['var_names_and_desc'])
                
                evidence_dict,success_ratio,evidence_ratio=search_causal_evidence(
                    env_step_info['latent_graph'],env_step_info['var_names_and_desc'],latent_desc,agent_id,logger=logger
                )
                evidence_ratio_lsit.append(evidence_ratio)
                evidence_strings.append(f"Agent {agent_id},casual_evidence:{evidence_dict},success_ratio:{success_ratio},evidence_ratio:{evidence_ratio}")
            logger.info(f"Evidence strings at t_env {t_env}: {evidence_strings}")
 

            # Generate commitment 
            commitment_text = self._generate_commitment(
                raw_observation_text, strategy, executor_responses,
                agent_info.get("group_repr"), agent_info.get("prompt_embeddings"),
                env_step_info['latent_graph'],env_step_info['var_names_and_desc']
            )
            # Save latent variable description
            save_latent_description(json.loads(commitment_text))

            
            # Get commitment embedding
            commitment_embedding = self.commitment_embedder.embed_commitments([commitment_text])
            
            agent_info.update({
                "strategy": strategy,
                "llm_responses": executor_responses,
                "confidence": confidence_lsit,
                "evidence_ratio_lsit":evidence_ratio_lsit,
                "executor_responses": executor_responses,
                "commitment": commitment_text,
                "commitment_text": commitment_text,
                "commitment_embedding": commitment_embedding
            })
        
        return chosen_actions, agent_info

    def forward(self, ep_batch: Any, t: int, test_mode: bool = False, train_mode: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through all agents.
        
        Args:
            ep_batch: Episode batch data
            t: Current timestep
            test_mode: Whether in test mode
            train_mode: Whether in training mode (for compatibility)
            
        Returns:
            Tuple of (agent_outputs, info_dict)
        """
        # Use train_mode if provided, otherwise use the inverse of test_mode
        actual_test_mode = test_mode if not train_mode else False
        
        # Build inputs for agents
        inputs, mask = self._build_inputs(ep_batch, t)
        
        # Forward pass through agents (LLMTransformerAgent) to obtain belief states, prompt embeddings, Q values, etc.
        agent_outs, hidden_states = self.agent(inputs, mask, test_mode=actual_test_mode)
        
        # Extract data from agent outputs
        # agent_outs contains outputs for all agents in the batch
        batch_size = ep_batch.batch_size
        
        # Extract and reshape outputs for each agent
        belief_states = agent_outs.get('belief_state', torch.zeros(batch_size * self.n_agents, self.args.belief_dim, device=self.device))# Ensure that if agent output misses belief_state, fill with zero tensor to avoid errors
        prompt_embeddings = agent_outs.get('prompt_embedding', torch.zeros(batch_size * self.n_agents, 2, device=self.device))
        q_values = agent_outs.get('q_value', torch.zeros(batch_size * self.n_agents, 1, device=self.device))# Check how the Q value is obtained
        
        # Reshape from (batch_size * n_agents, feature_dim) to (batch_size, n_agents, feature_dim)
        belief_states = belief_states.view(batch_size, self.n_agents, -1)
        prompt_embeddings = prompt_embeddings.view(batch_size, self.n_agents, -1)
        q_values = q_values.view(batch_size, self.n_agents, -1)
        
        # Generate group representation using BeliefEncoder
        group_representation = self.belief_encoder(belief_states)  # (batch, belief_dim)
        
        # Prepare outputs for compatibility with the rest of the system
        agent_outputs = q_values  # (batch_size, n_agents, 1)
        
        info_dict = {
            "belief_states": belief_states,
            "prompt_embeddings": prompt_embeddings,
            "q_values": q_values,
            "group_repr": group_representation,
            "hidden_states": hidden_states
        }
        
        return agent_outputs, info_dict

    def _get_strategy(self, question: str) -> str:
        """
        Generate strategy using coordinator LLM with token limit.
        
        Args:
            question: Input question
            
        Returns:
            Generated strategy (limited to 50 tokens)
        """
        # Check cache first
        if question in self.strategy_cache:
            return self.strategy_cache[question]
        
        # Generate improved strategy prompt with explicit token limit - NO DIRECT ANSWERS
        strategy_prompt = f"""You are a coordinator for mathematical problem-solving agents. Analyze this math problem and provide a clear solving strategy WITHOUT calculating the final answer.

Problem: {question}

REQUIREMENTS:
1. Your response must be EXACTLY 50 tokens or less
2. Provide step-by-step approach and methodology ONLY
3. DO NOT calculate numbers or provide the final answer
4. Focus on the solving process and required operations
5. Emphasize that final answer must be in \\boxed{{numerical_answer}} format

IMPORTANT: 
- Do NOT solve the problem yourself
- Only provide strategy and method
- Keep response within 50 tokens

Strategy (method only, no calculations):"""
        
        try:
            strategy = self.coordinator.generate_response(
                prompt=strategy_prompt,
                temperature=0.3,  # Lower temperature for more consistent strategies
                max_tokens=50  # Strict limit to prevent exceeding
            )# generate strategy
            
            # Log the generated strategy
            logger.info(f"📋 COORDINATOR STRATEGY: {strategy}")
            
            # Cache the strategy
            if len(self.strategy_cache) < self.max_cache_size:
                self.strategy_cache[question] = strategy
            
            return strategy
            
        except Exception as e:
            logger.warning(f"Failed to generate strategy: {e}")
            fallback_strategy = "Solve step by step: 1) Identify given values 2) Apply operations 3) Present answer in \\boxed{{}} format"
            logger.info(f"📋 COORDINATOR STRATEGY (FALLBACK): {fallback_strategy}")
            return fallback_strategy
    # TODO: generate final description for a single latent variable, needs modification
    def _generate_commitment(self, question: str, strategy: str, 
                           responses: List[str], group_repr: Optional[torch.Tensor] = None,
                           prompt_embeddings: Optional[torch.Tensor] = None,
                           latent_graph: List[Tuple[str,str]] = None,
                           var_names_and_desc: Dict = None) -> str:
        """
        Generate commitment using coordinator LLM with token limit.
        
        Args:
            question: Original question
            strategy: Generated strategy
            responses: Agent responses
            group_repr: Group representation tensor
            prompt_embeddings: Prompt embeddings tensor
            
        Returns:
            Generated commitment (limited to 50 tokens)
        """
        # Create cache key
        cache_key = f"{question}_{strategy}_{hash(tuple(responses))}"
        
        # Check cache first
        if cache_key in self.commitment_cache:
            return self.commitment_cache[cache_key]
        
        # Format responses for display
        formatted_responses = self._format_responses(responses)
        

        
        # Generate improved commitment prompt with explicit token limit
        commitment_prompt = f"""You are a coordinator. Review the given latent variable descriptions and 
their associated causal graph evidence. Based on the evidence, choose the most accurate description with the highest number of "true" results.
If all results are "false," infer the description based on the original causal graph and known variables. Keep your response within 20 tokens.

Agent Solutions:
{formatted_responses}


Original Causal Graph:
{latent_graph}

Known Variables:
{var_names_and_desc}

REQUIREMENTS:
1. Your response must be EXACTLY 20 tokens or less
2. Choose the most accurate latent variable description
3. Derive the final latent variable description. 


Output format (JSON):
You will return a single JSON dictionary mapping the specified latent variable to its description.
Example:{{
  "L1": "General predisposition to pain sensitivity based on genetic and physiological factors."
}}

IMPORTANT: Keep your commitment concise and within 50 tokens. Do not exceed this limit.

Final Answer (max 20 tokens):"""
        
        try:
            commitment = self.coordinator.generate_response(
                prompt=commitment_prompt,
                temperature=0.1,  # Very low temperature for precise commitments
                max_tokens=50  # Strict limit to prevent exceeding
            )
            
            # Validate and fix boxed answer format - remove
            # commitment = self._ensure_boxed_format(commitment)
            
            # Log the commitment - the coordinator generates the final latent variable description and caches it
            logger.info(f"🎯 COORDINATOR COMMITMENT: {commitment}")
            
            # Cache the commitment
            if len(self.commitment_cache) < self.max_cache_size:
                self.commitment_cache[cache_key] = commitment
            
            return commitment
            
        except Exception as e:
            logger.warning(f"Failed to generate commitment: {e}")
            logger.info(f"🎯 COORDINATOR COMMITMENT (FALLBACK): {fallback_commitment}")
            return fallback_commitment

    def _ensure_boxed_format(self, commitment: str) -> str:
        """
        Ensure commitment contains properly formatted boxed answer.
        
        Args:
            commitment: Generated commitment text
            
        Returns:
            Commitment with proper boxed format
        """
        import re
        
        # Check if already has boxed format
        if "\\boxed{" in commitment and "}" in commitment:
            # Extract and clean the boxed content
            boxed_match = re.search(r'\\boxed\{([^}]*)\}', commitment)
            if boxed_match:
                boxed_content = boxed_match.group(1).strip()
                # Clean up content - keep only numerical answer
                clean_content = re.sub(r'[^0-9\.\-]', '', boxed_content)
                if clean_content:
                    # Replace with cleaned content
                    commitment = re.sub(r'\\boxed\{[^}]*\}', f'\\boxed{{{clean_content}}}', commitment)
                    return commitment
        
        # If no boxed format or invalid format, try to extract numerical answer
        numbers = re.findall(r'-?\d+(?:\.\d+)?', commitment)
        if numbers:
            # Use the last number found as the answer
            answer = numbers[-1]
            if "\\boxed{" in commitment:
                # Replace existing boxed content
                commitment = re.sub(r'\\boxed\{[^}]*\}', f'\\boxed{{{answer}}}', commitment)
            else:
                # Add boxed format
                commitment += f" \\boxed{{{answer}}}"
        else:
            # No numbers found, add fallback
            if "\\boxed{" not in commitment:
                commitment += " \\boxed{0}"
        
        return commitment

    def _format_responses(self, responses: List[str]) -> str:
        """Format agent responses for commitment generation."""
        formatted = []
        for i, response in enumerate(responses):
            formatted.append(f"Agent {i+1}: {response}")
        return "\n".join(formatted)

    def _get_input_shape(self, scheme: Dict) -> int:
        """
        Get input shape for agents based on observation scheme.
        
        Args:
            scheme: Data scheme dictionary
            
        Returns:
            Input shape for agents
        """
        # For tokenized observations, use the vocabulary size
        if hasattr(self.args, 'vocab_size'):
            return self.args.vocab_size
        else:
            # Default vocabulary size (GPT2)
            return 50257

    def _get_default_actions(self, bs: slice, 
                           avail_actions: torch.Tensor) -> torch.Tensor:
        """Get default actions when agent forward fails."""
        batch_size = avail_actions.shape[0]
        # Return random valid actions
        random_actions = torch.randint(0, 2, (batch_size, self.n_agents), device=self.device)
        return random_actions

    def _get_default_outputs(self, ep_batch: Any) -> torch.Tensor:
        """Get default outputs when forward pass fails."""
        batch_size = ep_batch.batch_size
        return torch.zeros(batch_size, self.n_agents, self.args.n_actions, device=self.device)

    def cuda(self):
        """Move all components to CUDA."""
        self.agent.cuda()
        self.coordinator.cuda()
        self.belief_encoder.cuda()
        self.commitment_embedder.cuda()

    def save_models(self, path: str):
        """Save all model components."""
        self.agent.save_models(path)
        # Additional saving logic for other components if needed

    def load_models(self, path: str):
        """Load all model components."""
        self.agent.load_models(path)
        # Additional loading logic for other components if needed

    def _create_minimal_tokenizer(self):
        """Create a minimal tokenizer."""
        # Create a simple character-level tokenizer as fallback
        class MinimalTokenizer:
            def __init__(self):
                # Create a basic vocabulary
                self.vocab = {chr(i): i for i in range(32, 127)}  # ASCII printable characters
                self.vocab.update({'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3})
                self.pad_token = '[PAD]'
                self.eos_token = '[EOS]'
                self.pad_token_id = 0
                self.eos_token_id = 3
                self.vocab_size = len(self.vocab)
                
            def __call__(self, text, max_length=None, padding=True, truncation=True, return_tensors="pt"):
                if isinstance(text, str):
                    text = [text]
                
                # Simple tokenization by character
                tokenized = []
                for t in text:
                    tokens = [self.vocab.get(c, 1) for c in t[:max_length-1 if max_length else None]]
                    tokens.append(3)  # EOS token
                    
                    if max_length and padding:
                        if len(tokens) < max_length:
                            tokens.extend([0] * (max_length - len(tokens)))  # PAD tokens
                        tokens = tokens[:max_length]
                    
                    tokenized.append(tokens)
                
                if return_tensors == "pt":
                    import torch
                    return {"input_ids": torch.tensor(tokenized)}
                return tokenized
                
            def decode(self, token_ids, skip_special_tokens=True):
                # Simple decode implementation
                if hasattr(token_ids, 'tolist'):
                    token_ids = token_ids.tolist()
                
                text = ""
                reverse_vocab = {v: k for k, v in self.vocab.items()}
                for token_id in token_ids:
                    char = reverse_vocab.get(token_id, '[UNK]')
                    if skip_special_tokens and char in ['[PAD]', '[UNK]', '[BOS]', '[EOS]']:
                        continue
                    text += char
                return text
        
        logger.warning("Using minimal character-level tokenizer - this may affect performance")
        return MinimalTokenizer()

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to a specified token limit
        
        Args:
            text: the text to be truncated
            max_tokens: maximum number of tokens
            
        Returns:
            Truncated text
        """
        try:
            # Tokenize the text
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # If the number of tokens exceeds the limit, perform truncation
            if len(tokens) > max_tokens:
                # Truncate the token sequence
                truncated_tokens = tokens[:max_tokens]
                # Decode back to text
                truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                
                logger.debug(f"Truncated text from {len(tokens)} to {len(truncated_tokens)} tokens")
                return truncated_text
            else:
                return text
                
        except Exception as e:
            logger.warning(f"Error during token truncation: {e}")
            # If tokenizer fails, use simple word truncation as fallback
            words = text.split()
            if len(words) > max_tokens:
                return " ".join(words[:max_tokens])
            return text