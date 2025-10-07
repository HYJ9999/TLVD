import logging
import gym
from gym import spaces
import numpy as np
from datasets import load_dataset, Dataset, IterableDataset
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
import re
import torch
import torch.nn.functional as F
from utils.utils import edge_extraction, replace_observed_variables_desc, group_edges_by_latent
import json
from collections import defaultdict


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

class HuggingFaceDatasetEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, **kwargs):
        super().__init__()
        
        # self.dataset_path = kwargs.get("hf_dataset_path", "gsm8k")
        self.dataset_path = kwargs.get("graph_path", "./mac_collab/data/huaxi33_alpha0.003_rtscale1_N20.dot")
        self.var_desc_path = kwargs.get("var_desc_path", "./mac_collab/data/var_descriptions.json")
        self.latents_desc_path = kwargs.get("latents_desc_path", "./mac_collab/data/hidden_var_descriptions.json")
        
        self.dataset_config_name = kwargs.get("hf_dataset_config_name", None)
        self.dataset_split = kwargs.get("dataset_split", "train")
        self.is_streaming = kwargs.get("dataset_streaming", False)
        self.use_random_sampling = kwargs.get("use_random_sampling", False)  # Add random sampling option
        self.use_dataset_episode = kwargs.get("use_dataset_episode", False)  # Dataset-level episode option
        
        self.question_field = kwargs.get("latent_field_name", "latent")
        self.latent_graph_field = kwargs.get("latent_graph_field_name", "latent_graph") # latent variable related graph
        self.var_names_and_desc=kwargs.get("var_names_and_desc", "var_names_and_desc") # variable descriptions
        # For reward calculation, if needed directly in env
        self.reward_args = kwargs.get("reward_config", {})

        try:

            self.dataset=self.load_var_desc()

            if self.is_streaming:
                self.dataset_iterator = iter(self.dataset)
                # For IterableDataset, we can't easily get the length.
                # We might need a max_episodes arg from config for termination in streaming mode.
                logger.info(f"Loaded IterableDataset: {self.dataset_path}, split: {self.dataset_split}")
            else:
                # self.dataset_list = list(self.dataset) # Convert to list for easier iteration and shuffling if needed
                #for test
                self.dataset_list =self.dataset#   random.sample(list(self.dataset), 1)
                self.dataset_iterator = None # Will be created in reset
                self.current_data_idx = -1
                self.num_samples = len(self.dataset_list)
                logger.info(f"Loaded Dataset: {self.dataset_path}, split: {self.dataset_split}, num_samples: {self.num_samples}")
                

                if self.use_random_sampling and not self.use_dataset_episode:
                    random.shuffle(self.dataset_list)
                    logger.info("Dataset shuffled for random sampling")

        except Exception as e:
            logger.error(f"Failed to load dataset '{self.dataset_path}' (config: {self.dataset_config_name}, split: {self.dataset_split}): {e}")
            raise

        self.max_question_length = kwargs.get("max_question_length", 1024)
        self.max_answer_length = kwargs.get("max_answer_length", 1024) # For action space

        # Define action space - what the agent "outputs" to the environment
        # For LLMs, this is typically the generated text.
        # Using gym.spaces.Text requires gym version 0.26+
        # As a placeholder, or if direct text passing is used, can be simplified.
        self.action_space = spaces.Text(max_length=self.max_answer_length)

        # Define observation space - what the agent "sees"
        # This will be the question text.
        self.observation_space = spaces.Text(max_length=self.max_question_length)
        
        # Current sample from the dataset
        self.current_sample: Optional[Dict] = None
        self.current_question: Optional[str] = None
        self.current_ground_truth_answer: Optional[str] = None
        self.episode_count = 0  
        

        if self.use_dataset_episode:
            self.step_count = 0  
            self.episode_limit = self.num_samples if not self.is_streaming else 1000  # Use dataset size as episode limit
            self.current_episode_samples = [] 
            self.episode_results = []  
        else:
            # Episode specifics (each question is an episode)
            self.episode_length = 0 # Steps within current episode (always 1)
            self.episode_limit = 1

    def load_var_desc(self) -> List[Dict]:


        with open(self.var_desc_path, 'r') as f:
            var_names_and_desc = json.load(f)

        edges = edge_extraction(self.dataset_path)
        

        latent_edges_grouped = group_edges_by_latent(edges)
        

        result = []
        for latent, edges in latent_edges_grouped.items():
            result.append({
                "latent": latent,
                "latent_graph": edges,
                "var_names_and_desc":var_names_and_desc
            })
            

        result.sort(key=lambda x: int(re.findall(r"\d+", x["latent"])[0]))
        
        return result
    
    def _get_next_sample(self) -> Optional[Dict]:
        if self.is_streaming:
            try:
                return next(self.dataset_iterator)
            except StopIteration:
                logger.info("Streaming dataset iterator exhausted.")
                return None
        else:
            if self.use_dataset_episode:

                self.current_data_idx += 1
                if self.current_data_idx < self.num_samples:
                    return self.dataset_list[self.current_data_idx]
                else:
                    logger.info("Dataset-level episode completed: all samples processed.")
                    return None
            elif self.use_random_sampling:

                if self.num_samples > 0:
                    random_idx = random.randint(0, self.num_samples - 1)
                    sample = self.dataset_list[random_idx]
                    logger.debug(f"Random sampling: selected index {random_idx}")
                    return sample
                else:
                    logger.info("Dataset is empty.")
                    return None
            else:

                self.current_data_idx += 1
                logger.info(f"self.current_data_idx:{self.current_data_idx}")
                logger.info(f"self.num_samples:{self.num_samples}")
                if self.current_data_idx < self.num_samples:
                    return self.dataset_list[self.current_data_idx]
                else:
                    logger.info("Non-streaming dataset iterator exhausted.")
                    return None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed) # Gym 0.26+
        

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if self.use_dataset_episode:

            self.current_data_idx = -1  
            self.step_count = 0
            self.current_episode_samples = []
            self.episode_results = []
            self.episode_count += 1
            logger.info(f"Starting dataset-level Episode {self.episode_count}: will process {self.num_samples} samples")
        else:

            if not self.is_streaming:
                logger.info(f"self.current_data_idx >= self.num_samples - 1:{self.current_data_idx >= self.num_samples - 1}")
                if self.dataset_iterator is None and self.current_data_idx >= self.num_samples - 1:

                    if not self.use_random_sampling:
                        self.current_data_idx = -1

        self.current_sample = self._get_next_sample()
        
        if self.current_sample is None:
            # Handle dataset exhaustion, e.g., by raising an error or returning a special state
            # For now, let's raise an error to make it explicit during development.
            # In a long run, might want to loop the dataset or have a max_episodes from config.
            raise StopIteration("Dataset exhausted. Implement looping or max_episode limit if needed.")
        
        self.current_question = str(self.current_sample.get(self.question_field, ""))# current latent variable
        # self.current_ground_truth_answer = str(self.current_sample.get(self.latent_graph_field, ""))# current related graph
        self.latent_graph = str(self.current_sample.get(self.latent_graph_field, ""))# current related graph
        self.known_var_desc=self.current_sample.get(self.var_names_and_desc, "")# current known variable descriptions


        
        if not self.use_dataset_episode:
            self.episode_length = 0
            self.episode_count += 1
        
        # Log question changes for debugging
        if self.use_dataset_episode:
            question_preview = self.current_question[:100] + "..." if len(self.current_question) > 100 else self.current_question
            logger.info(f"Episode {self.episode_count}, Step {self.step_count + 1}/{self.num_samples}: {question_preview}")
        else:
            question_preview = self.current_question[:100] + "..." if len(self.current_question) > 100 else self.current_question
            logger.info(f"Episode {self.episode_count}: New Latent Variable - {question_preview}")
        
        # Observation is the question text
        # Preprocess if necessary (e.g., tokenization if obs_space was Box)
        # For now, passing raw text, MAC needs to handle it.
        # TODO: combine several fields into a paragraph as observation; append known latent variable description later
        self.latent_var_desc=None
        observation = f"Latent variable:{self.current_question}\nLatent_var_desc:{self.latent_var_desc}\nKnown_var_desc:{self.known_var_desc}\nCausal graph:{self.latent_graph}"  # self.current_question # observation is the question itself
        
        info = {"sample": self.current_sample} # Pass the whole sample for potential use in reward or logging
        # logger.info(f"observation:{observation}")
        # logger.info(f"info:{info}")
        return observation, info

    def step(self, action: Any, extra_info: Optional[Dict[str, Any]] = None) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: The primary action (final answer string from coordinator/agent)
            extra_info: Additional information for reward calculation including:
                - 'agent_responses': List[str] - Individual agent responses
                - 'commitment_text': str - Coordinator's commitment text
                - 'agent_log_probs': Optional[List[float]] - Token log probabilities for AL reward
                - 'prompt_embeddings': Optional[torch.Tensor] - Agent prompt embeddings
                - 'belief_states': Optional[torch.Tensor] - Agent belief states
        """
        if self.current_sample is None:
            raise RuntimeError("step() called before reset() or after dataset exhaustion.")

        if self.use_dataset_episode:
            self.step_count += 1
        else:
            self.episode_length += 1
        
        # Extract primary action and additional info
        if isinstance(action, dict):
            llm_answer_str = str(action.get("answer", ""))
            if extra_info is None:
                extra_info = action  # Use action dict as extra_info if not provided separately
        else:
            llm_answer_str = str(action)
            
        if extra_info is None:
            extra_info = {}


        logger.info("=" * 80)
        logger.info(f"🔍 QUESTION: {self.current_question}")
        logger.info("=" * 80)
        
        # Strategy will be logged by MAC
        
        # Executor responses will be logged by MAC
        
        # Coordinator commitment will be logged by MAC
        
        # logger.info(f"📖 GROUND TRUTH: {self.current_ground_truth_answer}")
        logger.info("=" * 80)

        # --- Reward Calculation ---
        # Task-Specific (TS) reward: based on correctness; uses coordinate LLM output and ground truth. Not calculated here; remove.

        # is_correct = self._evaluate_answer(llm_answer_str, self.current_ground_truth_answer)
        is_correct=True
        # TODO: Uncertainty reward: based on each agent's confidence and response similarity
        agent_responses = extra_info.get('agent_responses', [])
        

        reward_ur = self._calculate_uncertainty_reward(extra_info)
        reward_ur = sum(reward_ur) / len(reward_ur) if reward_ur else 0.0
        
        # Action Likelihood (AL) reward
        reward_al = self._calculate_action_likelihood_reward(extra_info)
        reward_al = sum(reward_al) / len(reward_al) if reward_al else 0.0
        
        # Collaborative Contribution (CC) reward
        reward_cc = self._calculate_collaborative_contribution_reward(
            llm_answer_str, extra_info, is_correct
        )
        reward_cc = sum(reward_cc) / len(reward_cc) if reward_cc else 0.0
        
       
        reward_er = extra_info.get('evidence_ratio_list') if extra_info.get('evidence_ratio_list') else [0.0] * len(agent_responses)
        reward_er = sum(reward_er) / len(reward_er) if reward_er else 0.0
        
       
        ur_weight = getattr(self.reward_args, 'ur_weight', 0.2)
        al_weight = getattr(self.reward_args, 'al_weight', 0.2)
        cc_weight = getattr(self.reward_args, 'cc_weight', 0.3)
        er_weight = getattr(self.reward_args, 'er_weight', 0.3)
        
        total_reward = ur_weight * reward_ur + al_weight * reward_al  + cc_weight * reward_cc + er_weight * reward_er
        
       
        logger.info(f"🎯 REWARD BREAKDOWN:")
        logger.info(f"   UR (Uncertainty Reward): {ur_weight:.3f} * {reward_ur:.1f} = {ur_weight * reward_ur:.3f}")
        logger.info(f"   AL (Action Likelihood): {al_weight:.3f} * {reward_al:.1f} = {al_weight * reward_al:.3f}")
        logger.info(f"   CC (Collaborative): {cc_weight:.3f} * {reward_cc:.1f} = {cc_weight * reward_cc:.3f}")
        logger.info(f"   ER (Evidence Reliability): {er_weight:.3f} * {reward_er:.1f} = {er_weight * reward_er:.3f}")
        logger.info(f"   TOTAL REWARD: {total_reward:.3f}")
        logger.info("=" * 80)
        
        
        if self.use_dataset_episode:
            
            step_result = {
                "question": self.current_question,
                "ground_truth": self.current_ground_truth_answer,
                "llm_answer": llm_answer_str,
                "is_correct": is_correct,
                "reward_ur": reward_ur,
                "reward_al": reward_al,
                "reward_cc": reward_cc,
                "reward_er":reward_er,
                "total_reward": total_reward
            }
            self.current_episode_samples.append(self.current_sample)
            self.episode_results.append(step_result)
            
          
            terminated = (self.step_count >= self.num_samples)
            
            if not terminated:
                
                self.current_sample = self._get_next_sample()
                if self.current_sample is None:
                    terminated = True
                else:
                    self.current_question = str(self.current_sample.get(self.question_field, ""))
                    self.current_ground_truth_answer = str(self.current_sample.get(self.answer_field, ""))
            
            if terminated:
                
                total_correct = sum(1 for r in self.episode_results if r["is_correct"])
                accuracy = total_correct / len(self.episode_results) if self.episode_results else 0.0
                avg_reward = sum(r["total_reward"] for r in self.episode_results) / len(self.episode_results) if self.episode_results else 0.0
                
                logger.info(f"📊 DATASET-LEVEL EPISODE {self.episode_count} COMPLETED:")
                logger.info(f"   Total samples: {len(self.episode_results)}")
                logger.info(f"   Correct answers: {total_correct}")
                logger.info(f"   Accuracy: {accuracy:.3f}")
                logger.info(f"   Average reward: {avg_reward:.3f}")
                logger.info("=" * 80)
                
                next_observation = ""  
            else:
                next_observation = self.current_question  
        else:
            
            terminated = True
            next_observation = ""  # Placeholder

        truncated = False # Not typically used if episode length is fixed at 1 or based on dataset size
        
        info = {
            "is_correct": is_correct,
            "reward_ur": reward_ur,
            "reward_al": reward_al,
            "reward_cc": reward_cc,
            "reward_er": reward_er,
            "llm_answer": llm_answer_str,
            "ground_truth_answer": ''#self.current_ground_truth_answer
        }
        
       
        if self.use_dataset_episode:
            info.update({
                "step_count": self.step_count,
                "total_steps": self.num_samples,
                "progress": self.step_count / self.num_samples
            })
            
            if terminated:
                
                total_correct = sum(1 for r in self.episode_results if r["is_correct"])
                info.update({
                    "episode_accuracy": total_correct / len(self.episode_results) if self.episode_results else 0.0,
                    "episode_avg_reward": sum(r["total_reward"] for r in self.episode_results) / len(self.episode_results) if self.episode_results else 0.0,
                    "total_samples_processed": len(self.episode_results)
                })
        
        return next_observation, total_reward, terminated, truncated, info

    def _extract_boxed_content(self, text: str) -> Optional[str]:
        """Extracts content from \\boxed{} with improved fallback mechanisms."""
        if not isinstance(text, str): # Ensure text is a string
            return None
        
        # Primary: Look for \\boxed{content}
        match = re.search(r"\\boxed\{([\s\S]*?)\}", text)
        if match:
            content = match.group(1).strip()
            return content if content else None
        
        # Fallback 1: Look for boxed{content} without backslash
        match = re.search(r"boxed\{([\s\S]*?)\}", text)
        if match:
            content = match.group(1).strip()
            logger.info(f"Found 'boxed{{}}' without backslash: {content}")
            return content if content else None
        
        # Fallback 2: Look for the last number in the text (often the final answer)
        numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        last_number_candidate = numbers[-1] if numbers else None
        
        # Fallback 3: Look for "answer is" patterns, but prefer last number if found
        patterns = [
            r"(?:answer is|answer:|final answer is|final answer:|the answer is)\s*([+-]?\d+(?:\.\d+)?)",
            r"(?:therefore|thus|so)\s*[^0-9]*([+-]?\d+(?:\.\d+)?)",
            r"(?:equals|=)\s*([+-]?\d+(?:\.\d+)?)",
        ]
        
        pattern_matches = []
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                pattern_matches.append(match.group(1))
        
        # Strategy: If we have both pattern matches and a last number, 
        # prefer the last number if it appears in the later part of the text
        if last_number_candidate and pattern_matches:
            # Check if the last number appears after the pattern matches
            last_number_pos = text.rfind(last_number_candidate)
            pattern_positions = []
            for pattern_match in pattern_matches:
                pos = text.rfind(pattern_match)
                if pos != -1:
                    pattern_positions.append(pos)
            
            # If last number appears after pattern matches, prefer it
            if pattern_positions and last_number_pos > max(pattern_positions):
                logger.info(f"Using last number as it appears after pattern matches: {last_number_candidate}")
                return last_number_candidate
            elif pattern_matches:
                logger.info(f"Using pattern match: {pattern_matches[0]}")
                return pattern_matches[0]
        
        # If only pattern matches exist
        if pattern_matches:
            logger.info(f"Using pattern match: {pattern_matches[0]}")
            return pattern_matches[0]
        
        # If only last number exists
        if last_number_candidate:
            logger.info(f"Using last number in text as fallback: {last_number_candidate}")
            return last_number_candidate
        
        logger.warning(f"No answer found in text: {text[:100]}...")
        return None

    def _normalize_number_string(self, s: Optional[str]) -> Optional[str]:
        """Normalizes a string potentially representing a number."""
        if s is None:
            return None
        # Remove commas used as thousand separators
        s_no_commas = s.replace(",", "")
        # Remove trailing ".0" or ".00" etc. to treat 123.0 as 123 for int comparison
        # but keep 123.5 as 123.5
        if '.' in s_no_commas:
            parts = s_no_commas.split('.')
            if len(parts) == 2 and all(c == '0' for c in parts[1]):
                return parts[0] # Return only integer part if fractional part is all zeros
        return s_no_commas

    def _evaluate_answer(self, llm_answer: str, ground_truth_answer: str) -> bool:
        logger.debug(f"Evaluating LLM Answer: '{llm_answer}' vs Ground Truth: '{ground_truth_answer}'")

        llm_boxed_content = self._extract_boxed_content(llm_answer)
        gt_boxed_content = self._extract_boxed_content(ground_truth_answer)

        logger.debug(f"Boxed Content - LLM: '{llm_boxed_content}', GT: '{gt_boxed_content}'")

        #
        if llm_boxed_content is None and gt_boxed_content is None:
            logger.info("Both answers lack \\boxed{} format, attempting direct text comparison")
           
            llm_numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", llm_answer)
            gt_numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", ground_truth_answer)
            
            if llm_numbers and gt_numbers:
                llm_boxed_content = llm_numbers[-1]
                gt_boxed_content = gt_numbers[-1]
                logger.info(f"Extracted numbers - LLM: '{llm_boxed_content}', GT: '{gt_boxed_content}'")
            else:
                logger.info(f"Evaluation failed: Unable to extract numerical answers. LLM: '{llm_answer[:100]}...', GT: '{ground_truth_answer[:100]}...'")
                return False

        
        if llm_boxed_content is None or gt_boxed_content is None:
            logger.info(f"Evaluation failed: Inconsistent answer formats. LLM boxed: '{llm_boxed_content}', GT boxed: '{gt_boxed_content}'")
            logger.info(f"Full answers - LLM: '{llm_answer[:150]}...', GT: '{ground_truth_answer[:150]}...'")
            return False

        # Normalize the string content from \boxed{} before attempting float conversion or string comparison
        norm_llm_content = self._normalize_number_string(llm_boxed_content)
        norm_gt_content = self._normalize_number_string(gt_boxed_content)
        
        logger.debug(f"Normalized Boxed Content - LLM: '{norm_llm_content}', GT: '{norm_gt_content}'")

        if norm_llm_content is None or norm_gt_content is None: # Should not happen if _extract_boxed_content returned non-None
             return False

        try:
            # Attempt to convert both to floats for numerical comparison
            llm_val = float(norm_llm_content)
            gt_val = float(norm_gt_content)

            # Check for near-equality
            if abs(llm_val - gt_val) < 1e-5:
                logger.info(f"✅ Correct answer: {llm_val} == {gt_val}")
                return True
            else:
                logger.info(f"❌ Numeric mismatch: LLM val {llm_val} vs GT val {gt_val}")
                return False
        except ValueError:
            # If conversion to float fails, fall back to string comparison
            logger.debug(f"ValueError converting to float. Comparing normalized strings: '{norm_llm_content}' vs '{norm_gt_content}'")
            if norm_llm_content == norm_gt_content:
                logger.info(f"✅ Correct answer (string match): '{norm_llm_content}'")
                return True
            else:
                # Last resort: compare the original (just stripped) boxed content
                if llm_boxed_content.strip() == gt_boxed_content.strip():
                    logger.info(f"✅ Correct answer (original content match): '{llm_boxed_content.strip()}'")
                    return True
                logger.info(f"❌ String mismatch after float conversion failed. LLM: '{norm_llm_content}', GT: '{norm_gt_content}'")
                return False

    def _calculate_action_likelihood_reward(self, extra_info: Dict[str, Any]) -> List[float]:
        """Compute the action likelihood reward r^AL for each agent
        Based on the following factors:
        1. Quality of each agent's response
        2. Similarity of each response to the commitment
        3. If the API fails or the response is invalid, return 0.0 for the corresponding agent
        Returns a list of rewards for each agent
        """
        try:
            
            agent_responses = extra_info.get('agent_responses', [])
            commitment_text = extra_info.get('commitment_text', '')
            
           
            if not agent_responses or not commitment_text:
                return [0.0] * len(agent_responses) if agent_responses else [0.0]
            
           
            error_indicators = ['Error: Could not generate response', 'API Error', 'HTTP error', 'Failed to generate']
            
            
            al_rewards = []
            
            
            if any(error in str(commitment_text) for error in error_indicators):
                return [0.0] * len(agent_responses)
            
           
            for response in agent_responses:
                
                if any(error in str(response) for error in error_indicators) or not str(response).strip():
                    al_rewards.append(0.0)
                    continue
                
               
                text_similarity = 0.0
                if len(response.strip()) >= 10:
                    
                    response_latent_desc = {k: v for k, v in json.loads(response).items() if k != "confidence"}
                    text_similarity = self._calculate_text_similarity(json.dumps(response_latent_desc), commitment_text)
                
               
                response_length = len(response.strip())
                if 20 <= response_length <= 500:  
                    length_score = 1.0
                elif 10 <= response_length <= 1000:  
                    length_score = 0.7
                else:  
                    length_score = 0.3
                
                
                agent_reward = 0.7 * text_similarity + 0.3 * length_score
                al_rewards.append(min(1.0, max(0.0, agent_reward)))
            
            return al_rewards
            
        except Exception as e:
            logger.warning(f"Error calculating AL reward: {e}")
            
            return [0.0] * len(agent_responses) if agent_responses else [0.0]

    def _calculate_collaborative_contribution_reward(self, final_answer: str, 
                                                   extra_info: Dict[str, Any], 
                                                   is_correct: bool) -> List[float]:
        """Compute the collaboration contribution reward r^CC for each agent
        Based on the following heuristic rules:
        1. Base reward: each agent gets the same base reward
        2. Uniqueness reward: based on how unique the agent's response is
        3. Consistency reward: based on the similarity between the agent's response and the commitment
        4. If the API fails or the response is invalid, return 0.0 for the corresponding agent
        """
        try:
            agent_responses = extra_info.get('agent_responses', [])
            commitment_text = extra_info.get('commitment_text', '')
            
            
            error_indicators = ['Error: Could not generate response', 'API Error', 'HTTP error', 'Failed to generate']
            
            
            if (any(error in str(final_answer) for error in error_indicators) or 
                any(error in str(commitment_text) for error in error_indicators)):
                return [0.0] * len(agent_responses) if agent_responses else [0.0]
            
            #
            cc_rewards = []
            
            
            # base_reward = 0.3 #if is_correct else 0.0
            
            
            valid_responses_set = set()
            for response in agent_responses:
                if not any(error in str(response) for error in error_indicators):
                    valid_responses_set.add(response.strip().lower())
            
            
            for response in agent_responses:
                
                if any(error in str(response) for error in error_indicators) or not str(response).strip():
                    cc_rewards.append(0.0)
                    continue
                
                
                
                consistency_score = 0.0
                if len(response.strip()) >= 10:
                    
                    response_latent_desc = {k: v for k, v in json.loads(response).items() if k != "confidence"}
                    sim = self._calculate_text_similarity(json.dumps(response_latent_desc), commitment_text)
                    consistency_score = 0.8 * sim
                
               
                quality_reward = 0.0
                
                response_length = len(response.strip())
                if 5 <= response_length <= 20:
                    quality_reward += 0.1
                
                try:
                    json.loads(response.strip())
                    quality_reward += 0.1
                except (json.JSONDecodeError, TypeError):
                    pass  
                
                
                agent_reward = consistency_score + quality_reward
                cc_rewards.append(min(1.0, max(0.0, agent_reward)))
            
            return cc_rewards if cc_rewards else [0.0] * len(agent_responses)
            
        except Exception as e:
            logger.warning(f"Error calculating CC reward: {e}")
            return [0.0] * len(agent_responses) if agent_responses else [0.0]

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two pieces of text
        
        Args:
            text1: first text
            text2: second text
            
        Returns:
            float: similarity score (0-1)
        """
        if not isinstance(text1, str) or not isinstance(text2, str):
            return 0.0
            
        
        vectorizer = TfidfVectorizer()
        
        try:
            
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.0
            
    def _calculate_uncertainty_reward(self, extra_info: Dict[str, Any]) -> List[float]:
        """Compute the uncertainty reward for each agent
        
        Args:
            extra_info: dictionary containing:
                - agent_responses: list of agent responses
                - commitment_text: final commitment text
                - confidence: list of confidence levels for each agent
            
        Returns:
            List[float]: list of uncertainty rewards for each agent
        """
       
        confidence_list = extra_info.get('confidence_lsit', [])
        agent_responses = extra_info.get('agent_responses', [])
        commitment_text = extra_info.get('commitment_text', '')
        reward_ur = []

       
        if not agent_responses or not commitment_text:
            logger.warning("Missing required information for uncertainty reward calculation")
            return [0.0] * len(confidence_list) if confidence_list else [0.0]

        
        for i, response in enumerate(agent_responses):
            try:
                if not isinstance(response, str) or not response.strip():
                    reward_ur.append(0.0)
                    continue

                
                response_dict = json.loads(response)
                response_latent_desc = {k: v for k, v in response_dict.items() if k != "confidence"}
                
               
                response_similarity = self._calculate_text_similarity(json.dumps(response_latent_desc), commitment_text)
                
               
                confidence = confidence_list[i] if i < len(confidence_list) else 0.0
                
                
                agent_reward = -confidence * response_similarity
                reward_ur.append(agent_reward)
            
            except (json.JSONDecodeError, TypeError, Exception) as e:
                logger.warning(f"Error calculating uncertainty reward for agent {i}: {e}")
                reward_ur.append(0.0)
        
        
        if not reward_ur:
            reward_ur = [0.0] * len(confidence_list) if confidence_list else [0.0]
            
        return reward_ur

    def get_env_info(self) -> Dict[str, Any]:
        # This info is used by the runner to setup the batch scheme.
        return {
            "episode_limit": self.episode_limit,
            "n_actions": 1,  # Placeholder - action space is text
            "obs_shape": (self.max_question_length,), # For scheme vshape
            "state_shape": (1,), # Placeholder for scheme vshape
            # Any other info needed by the runner or learner
        }

    def render(self, mode='human'):
        if mode == 'human':
            if self.current_sample:
                print("-" * 30)
                print(f"Current Question: {self.current_question}")
                print(f"Ground Truth Answer: {self.current_ground_truth_answer}")
                print("-" * 30)
            else:
                print("No current sample to render. Call reset() first.")

    def close(self):
        # Clean up resources if any (e.g., closing file handles if dataset was local)
        logger.info("Closing HuggingFaceDatasetEnv.")
        pass

# Example Usage (for testing purposes):
if __name__ == '__main__':
    
    env_args_gsm8k = {
        "hf_dataset_path": "gsm8k",
        "hf_dataset_config_name": "main",
        "dataset_split": "test",
        "question_field_name": "question",
        "answer_field_name": "answer",
        "max_question_length": 1024,
        "max_answer_length": 200,
        "dataset_streaming": False, # For testing, non-streaming is easier
        "use_random_sampling": False, # Disable random sampling for deterministic testing
        "use_dataset_episode": False # Disable dataset-level episode for deterministic testing
    }
    


    # Test MATH dataset
    env_args_math = {
        "hf_dataset_path": "competition_math",
        "dataset_split": "test", # Using test split which is smaller
        "question_field_name": "problem",
        "answer_field_name": "solution",
        "max_question_length": 2048,
        "max_answer_length": 2048,
        "dataset_streaming": False,
        "use_random_sampling": False, # Disable random sampling for deterministic testing
        "use_dataset_episode": False # Disable dataset-level episode for deterministic testing
    }
    math_env = HuggingFaceDatasetEnv(**env_args_math)
    obs, info = math_env.reset()
    math_env.render()
    # Simulate a step for MATH
    # For MATH, answers are more complex, often with LaTeX. Evaluation is harder.
    # Example: ground truth might be "\\boxed{-\\frac{1}{2}}"
    # dummy_math_action = "The final answer is \\boxed{-\\frac{1}{2}}"
    dummy_math_action = info['sample'][math_env.answer_field] # Give correct answer
    next_obs, reward, terminated, truncated, step_info = math_env.step(dummy_math_action)
    print(f"LLM's Action: {dummy_math_action}")
    print(f"Reward: {reward}") # Should be 1.0
    print(f"Step Info: {step_info}")
    math_env.render()

    # Test an incorrect answer for MATH
    obs, info = math_env.reset()
    math_env.render()
    dummy_math_action_wrong = "The final answer is \\boxed{42}"
    next_obs, reward, terminated, truncated, step_info = math_env.step(dummy_math_action_wrong)
    print(f"LLM's Action (Wrong): {dummy_math_action_wrong}")
    print(f"Reward: {reward}") # Should be 0.0
    print(f"Step Info: {step_info}")
    math_env.render()