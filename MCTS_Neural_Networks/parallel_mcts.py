import numpy as np
import torch
import torch.multiprocessing as mp
from mcts import MCTS


class ParallelMCTS:
    """
    Parallel implementation of Monte Carlo Tree Search
    Uses multiple processes to speed up search
    """
    
    def __init__(self, model, game_env, num_processes=4, num_simulations=800, 
                 exploration_constant=1.0, dirichlet_noise=True, 
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 temperature=1.0, confidence_threshold=0.95,
                 root_noise=True):
        """
        Initialize the Parallel MCTS algorithm
        
        Args:
            model: Neural network model that provides policy and value predictions
            game_env: Game environment that implements the game rules
            num_processes: Number of parallel processes to use
            num_simulations: Number of simulations to run for each search
            exploration_constant: Controls exploration in UCB formula
            dirichlet_noise: Whether to add Dirichlet noise to the root node for exploration
            dirichlet_alpha: Alpha parameter for Dirichlet distribution
            dirichlet_epsilon: Weight of Dirichlet noise
            temperature: Temperature for action selection (higher = more exploration)
            confidence_threshold: Threshold for confidence-based pruning
            root_noise: Whether to add noise only to the root node
        """
        self.model = model
        self.game_env = game_env
        self.num_processes = num_processes
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.dirichlet_noise = dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.root_noise = root_noise
        
        # Make sure the model is in shared memory
        self.model.share_memory()
        
        # Create process pool
        self.pool = None
        
    def search(self, state, knowledge_embedding=None, time_step=0):
        """
        Run parallel MCTS algorithm to find the best action
        
        Args:
            state: Current game state
            knowledge_embedding: Optional knowledge embedding for state
            time_step: Current time step for temporal reasoning
            
        Returns:
            Action probabilities after search
        """
        # Convert state to string for consistency
        state_str = str(state)
        
        # Check if game is over
        if self.game_env.is_terminal(state):
            return None
        
        # Create process pool if not already created
        if self.pool is None:
            self.pool = mp.Pool(processes=self.num_processes)
        
        # Calculate simulations per process
        sims_per_process = self.num_simulations // self.num_processes
        
        # Create arguments for each process
        process_args = []
        for i in range(self.num_processes):
            # Only add Dirichlet noise to the first process if root_noise is True
            use_noise = self.dirichlet_noise if (not self.root_noise or i == 0) else False
            
            process_args.append((
                self.model,
                self.game_env,
                state,
                knowledge_embedding,
                time_step,
                sims_per_process,
                self.exploration_constant,
                use_noise,
                self.dirichlet_alpha,
                self.dirichlet_epsilon,
                self.confidence_threshold
            ))
        
        # Run MCTS in parallel
        results = self.pool.map(self._worker_search, process_args)
        
        # Combine results
        combined_visits = np.zeros(self.game_env.action_space_size())
        for visit_counts in results:
            combined_visits += visit_counts
        
        # Apply temperature
        if self.temperature > 0:
            combined_visits = combined_visits ** (1 / self.temperature)
            
        # Normalize
        action_probs = combined_visits / np.sum(combined_visits) if np.sum(combined_visits) > 0 else combined_visits
        
        return action_probs
    
    @staticmethod
    def _worker_search(args):
        """
        Worker function for parallel search
        
        Args:
            args: Tuple of arguments for search
            
        Returns:
            Visit counts for each action
        """
        (model, game_env, state, knowledge_embedding, time_step, 
         num_simulations, exploration_constant, dirichlet_noise, 
         dirichlet_alpha, dirichlet_epsilon, confidence_threshold) = args
        
        # Create MCTS instance
        mcts = MCTS(
            model=model,
            game_env=game_env,
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            dirichlet_noise=dirichlet_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            confidence_threshold=confidence_threshold
        )
        
        # Run search
        mcts.search(state, knowledge_embedding, time_step)
        
        # Extract visit counts for each action
        state_str = mcts._state_to_string(state)
        visit_counts = np.zeros(game_env.action_space_size())
        
        if state_str in mcts.children:
            for action in mcts.children[state_str]:
                action_str = mcts._state_action_to_string(state_str, action)
                visit_counts[action] = mcts.visit_count[action_str]
        
        return visit_counts
    
    def get_best_action(self, state, deterministic=False):
        """
        Get best action after search
        
        Args:
            state: Current state
            deterministic: Whether to choose the best action deterministically
            
        Returns:
            Best action
        """
        action_probs = self.search(state)
        
        if deterministic:
            # Choose action with highest probability
            return np.argmax(action_probs)
        else:
            # Sample from the distribution
            return np.random.choice(len(action_probs), p=action_probs)
    
    def close(self):
        """
        Close the process pool
        """
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None 