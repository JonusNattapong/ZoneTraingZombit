import numpy as np
import torch
from collections import defaultdict
import math


class MCTS:
    """
    Monte Carlo Tree Search with Neural Networks
    """

    def __init__(self, model, game_env, num_simulations=800, exploration_constant=1.0, 
                 dirichlet_noise=True, dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 temperature=1.0, confidence_threshold=0.95):
        """
        Initialize the MCTS algorithm
        
        Args:
            model: Neural network model that provides policy and value predictions
            game_env: Game environment that implements the game rules
            num_simulations: Number of simulations to run for each search
            exploration_constant: Controls exploration in UCB formula
            dirichlet_noise: Whether to add Dirichlet noise to the root node for exploration
            dirichlet_alpha: Alpha parameter for Dirichlet distribution
            dirichlet_epsilon: Weight of Dirichlet noise
            temperature: Temperature for action selection (higher = more exploration)
            confidence_threshold: Threshold for confidence-based pruning
        """
        self.model = model
        self.game_env = game_env
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.dirichlet_noise = dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        
        # Tree statistics
        self.visit_count = defaultdict(int)
        self.value_sum = defaultdict(float)
        self.children = {}
        self.policy = {}
        self.confidence = {}
        self.temporal_value = {}
        
        # For multi-level search
        self.difficulty_level = {}
        self.time_steps = {}
        
        # For knowledge integration
        self.knowledge_embeddings = {}

    def search(self, state, knowledge_embedding=None, time_step=0):
        """
        Run MCTS algorithm to find the best action
        
        Args:
            state: Current game state
            knowledge_embedding: Optional knowledge embedding for state
            time_step: Current time step for temporal reasoning
            
        Returns:
            Action probabilities after search
        """
        # Convert state to string for dictionary lookup
        state_str = self._state_to_string(state)
        
        # Store knowledge embedding and time step
        if knowledge_embedding is not None:
            self.knowledge_embeddings[state_str] = knowledge_embedding
        self.time_steps[state_str] = time_step
        
        # Check if game is over
        if self.game_env.is_terminal(state):
            return None
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(state)
            
        # Calculate visit count-based probabilities
        visit_counts = np.array([
            self.visit_count[self._state_action_to_string(state_str, action)]
            for action in range(self.game_env.action_space_size())
        ])
        
        # Apply temperature
        if self.temperature > 0:
            visit_counts = visit_counts ** (1 / self.temperature)
            
        # Normalize
        action_probs = visit_counts / np.sum(visit_counts)
        
        return action_probs
    
    def _simulate(self, state):
        """Run a single MCTS simulation"""
        search_path = []
        current_state = state
        state_str = self._state_to_string(current_state)
        
        # Selection: traverse tree until we find a leaf node
        while state_str in self.children and not self.game_env.is_terminal(current_state):
            # Get valid actions
            valid_actions = self.game_env.get_valid_actions(current_state)
            
            # Select action with highest UCB score
            action = self._select_action(state_str, valid_actions)
            
            # Record visit
            search_path.append((state_str, action))
            
            # Apply action
            current_state = self.game_env.step(current_state, action)
            state_str = self._state_to_string(current_state)
        
        # If leaf node is not terminal and not in tree, expand
        if not self.game_env.is_terminal(current_state) and state_str not in self.children:
            # Expansion and evaluation
            self._expand_node(current_state, state_str)
        
        # Get value for backpropagation
        if self.game_env.is_terminal(current_state):
            value = self.game_env.get_reward(current_state)
        else:
            value = self.value_sum[state_str] / max(1, self.visit_count[state_str])
        
        # Backpropagation
        self._backpropagate(search_path, value)
    
    def _expand_node(self, state, state_str):
        """Expand a leaf node and evaluate it"""
        # Get valid actions
        valid_actions = self.game_env.get_valid_actions(state)
        
        # Get knowledge embedding if available
        k_embedding = self.knowledge_embeddings.get(state_str, None)
        
        # Get time step
        time_step = self.time_steps.get(state_str, 0)
        
        # Evaluate state with the neural network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(self.game_env.get_encoded_state(state)).unsqueeze(0)
            
            if k_embedding is not None:
                # If we have knowledge embedding, concatenate it
                k_tensor = torch.FloatTensor(k_embedding).unsqueeze(0)
                policy, value, temporal_value, confidence, difficulty = self.model(state_tensor, k_tensor, time_step)
            else:
                # Otherwise, use standard forward pass
                policy, value = self.model(state_tensor)
                temporal_value = value
                confidence = torch.ones_like(value)
                difficulty = torch.ones((1, 3)) / 3  # Default: equally likely to be easy/medium/hard
        
        # Convert outputs to numpy
        policy = policy.squeeze(0).numpy()
        value = value.item()
        
        # Store temporal value, confidence and difficulty level
        self.temporal_value[state_str] = temporal_value.item()
        self.confidence[state_str] = confidence.item()
        self.difficulty_level[state_str] = np.argmax(difficulty.squeeze(0).numpy())
        
        # Mask invalid actions
        masked_policy = policy * valid_actions
        
        # Re-normalize
        sum_masked_policy = np.sum(masked_policy)
        if sum_masked_policy > 0:
            masked_policy /= sum_masked_policy
        else:
            # If all valid actions were masked, use uniform distribution
            masked_policy = valid_actions / np.sum(valid_actions)
        
        # Initialize children and policy
        self.children[state_str] = [a for a in range(len(valid_actions)) if valid_actions[a] == 1]
        self.policy[state_str] = masked_policy
        
        # Initialize node statistics
        self.visit_count[state_str] = 1
        self.value_sum[state_str] = value
    
    def _select_action(self, state_str, valid_actions):
        """Select action with highest UCB score"""
        # Get children actions
        children_actions = self.children[state_str]
        
        # If dirichlet noise is enabled and this is a root node (visit count is low)
        if self.dirichlet_noise and self.visit_count[state_str] < 2:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(children_actions))
            noisy_policy = (1 - self.dirichlet_epsilon) * self.policy[state_str] + self.dirichlet_epsilon * noise
        else:
            noisy_policy = self.policy[state_str]
        
        # Get confidence of the state
        confidence = self.confidence.get(state_str, 1.0)
        
        # Get difficulty level (0=easy, 1=medium, 2=hard)
        difficulty = self.difficulty_level.get(state_str, 1)
        
        # Adjust exploration constant based on difficulty
        adjusted_exploration = self.exploration_constant * (1.0 + 0.5 * difficulty)
        
        # Calculate exploration bonus factor based on confidence
        if confidence > self.confidence_threshold:
            # High confidence = less exploration
            exploration_factor = 0.5
        else:
            # Low confidence = more exploration
            exploration_factor = 1.5
        
        # Calculate UCB scores for all valid actions
        ucb_scores = np.zeros(len(valid_actions))
        for i, action in enumerate(range(len(valid_actions))):
            if valid_actions[action] == 0:
                continue
                
            # Create action string for lookup
            action_str = self._state_action_to_string(state_str, action)
            
            # If action not visited, assign infinite score
            if self.visit_count[action_str] == 0:
                ucb_scores[action] = float('inf')
                continue
            
            # Q-value: average value from this state-action
            q_value = self.value_sum[action_str] / self.visit_count[action_str]
            
            # Adjust Q-value with temporal value if available
            temporal_q_value = self.temporal_value.get(action_str, q_value)
            q_value = 0.8 * q_value + 0.2 * temporal_q_value
            
            # UCB formula
            ucb_score = q_value + adjusted_exploration * exploration_factor * noisy_policy[action] * \
                        math.sqrt(self.visit_count[state_str]) / (1 + self.visit_count[action_str])
            
            ucb_scores[action] = ucb_score
        
        # Select action with highest UCB score
        return np.argmax(ucb_scores)
    
    def _backpropagate(self, search_path, value):
        """Backpropagate the value through the search path"""
        # For each state-action pair in the search path
        for state_str, action in reversed(search_path):
            action_str = self._state_action_to_string(state_str, action)
            
            # Update visit count and value sum
            self.visit_count[action_str] += 1
            self.value_sum[action_str] += value
            
            # Update state visit count and value sum
            self.visit_count[state_str] += 1
            self.value_sum[state_str] += value
            
            # Flip value for alternating players in zero-sum games
            value = -value
    
    def _state_to_string(self, state):
        """Convert state to string for dictionary lookup"""
        return str(state)
    
    def _state_action_to_string(self, state_str, action):
        """Convert state-action pair to string for dictionary lookup"""
        return f"{state_str}_{action}"
    
    def get_best_action(self, state, deterministic=False):
        """Get best action after search"""
        action_probs = self.search(state)
        
        if deterministic:
            # Choose action with highest probability
            return np.argmax(action_probs)
        else:
            # Sample from the distribution
            return np.random.choice(len(action_probs), p=action_probs) 