import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
from collections import deque
import random

from mcts import MCTS
from parallel_mcts import ParallelMCTS


class SelfPlayDataset(Dataset):
    """
    Dataset for self-play training data
    """
    
    def __init__(self, states, policies, values):
        """
        Initialize dataset
        
        Args:
            states: List of encoded states
            policies: List of policy targets
            values: List of value targets
        """
        self.states = states
        self.policies = policies
        self.values = values
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.policies[idx]),
            torch.FloatTensor([self.values[idx]])
        )


class SelfPlayTrainer:
    """
    Trainer for MCTS with Neural Networks using self-play
    """
    
    def __init__(self, model, game_env, config=None, typhoon_model=None):
        """
        Initialize trainer
        
        Args:
            model: Neural network model
            game_env: Game environment
            config: Configuration dictionary
            typhoon_model: Optional typhoon-7b model for enhanced training
        """
        self.model = model
        self.game_env = game_env
        self.typhoon_model = typhoon_model
        
        # Default configuration
        self.config = {
            'num_iterations': 100,
            'num_self_play_games': 100,
            'num_mcts_simulations': 800,
            'temperature': 1.0,
            'batch_size': 256,
            'epochs_per_iteration': 10,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25,
            'buffer_size': 100000,
            'checkpoint_interval': 10,
            'checkpoint_dir': 'checkpoints',
            'use_parallel_mcts': True,
            'num_processes': 4,
            'eval_games': 20,
            'exploration_constant': 1.0,
            'confidence_threshold': 0.95,
            'value_loss_weight': 1.0,
            'policy_loss_weight': 1.0,
            'temperature_threshold': 30
        }
        
        # Update configuration if provided
        if config:
            self.config.update(config)
        
        # Create checkpoint directory
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=self.config['buffer_size'])
        
        # Create MCTS instance (will be used for training)
        if self.config['use_parallel_mcts']:
            self.mcts = ParallelMCTS(
                model=self.model,
                game_env=self.game_env,
                num_processes=self.config['num_processes'],
                num_simulations=self.config['num_mcts_simulations'],
                exploration_constant=self.config['exploration_constant'],
                dirichlet_noise=True,
                dirichlet_alpha=self.config['dirichlet_alpha'],
                dirichlet_epsilon=self.config['dirichlet_epsilon'],
                temperature=self.config['temperature'],
                confidence_threshold=self.config['confidence_threshold']
            )
        else:
            self.mcts = MCTS(
                model=self.model,
                game_env=self.game_env,
                num_simulations=self.config['num_mcts_simulations'],
                exploration_constant=self.config['exploration_constant'],
                dirichlet_noise=True,
                dirichlet_alpha=self.config['dirichlet_alpha'],
                dirichlet_epsilon=self.config['dirichlet_epsilon'],
                temperature=self.config['temperature'],
                confidence_threshold=self.config['confidence_threshold']
            )
        
        # Loss functions
        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.CrossEntropyLoss()
    
    def train(self):
        """
        Train the model using self-play
        """
        best_eval_score = float('-inf')
        
        for iteration in range(self.config['num_iterations']):
            start_time = time.time()
            
            print(f"Starting iteration {iteration+1}/{self.config['num_iterations']}")
            
            # 1. Self-play to generate training data
            self_play_data = self.self_play()
            
            # 2. Add data to replay buffer
            self.replay_buffer.extend(self_play_data)
            
            # 3. Train neural network
            self.train_neural_network()
            
            # 4. Evaluate model
            eval_score = self.evaluate()
            
            # 5. Save checkpoint
            if (iteration + 1) % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(iteration + 1)
                
            # Save best model
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                self.save_checkpoint(0, is_best=True)
            
            # Print iteration stats
            elapsed_time = time.time() - start_time
            print(f"Iteration {iteration+1} completed in {elapsed_time:.2f} seconds.")
            print(f"Evaluation score: {eval_score:.3f}")
            print(f"Replay buffer size: {len(self.replay_buffer)}")
            print("---------------------------------------------------")
        
        # Close MCTS if using parallel version
        if self.config['use_parallel_mcts']:
            self.mcts.close()
    
    def self_play(self):
        """
        Generate self-play games
        
        Returns:
            List of (state, policy, value) tuples
        """
        training_data = []
        games_played = 0
        
        while games_played < self.config['num_self_play_games']:
            # Initialize game state
            state = self.game_env.get_initial_state()
            game_memory = []
            
            # Play game
            turn = 0
            while not self.game_env.is_terminal(state):
                # Calculate temperature for this turn
                if turn < self.config['temperature_threshold']:
                    temperature = self.config['temperature']
                else:
                    temperature = 0.1  # Lower temperature for later moves
                
                # Set MCTS temperature
                if self.config['use_parallel_mcts']:
                    self.mcts.temperature = temperature
                else:
                    self.mcts.temperature = temperature
                
                # Get action probabilities from MCTS
                action_probs = self.mcts.search(state, time_step=turn)
                
                # Store state and MCTS probabilities
                encoded_state = self.game_env.get_encoded_state(state)
                game_memory.append((encoded_state, action_probs))
                
                # Choose action based on MCTS probabilities
                action = np.random.choice(len(action_probs), p=action_probs)
                
                # Apply action
                state = self.game_env.step(state, action)
                turn += 1
            
            # Game over, calculate reward
            reward = self.game_env.get_reward(state)
            
            # Add game data to training data with reward as value target
            for encoded_state, action_probs in game_memory:
                training_data.append((encoded_state, action_probs, reward))
                # Flip reward for alternating players in zero-sum games
                reward = -reward
            
            games_played += 1
            print(f"Self-play game {games_played}/{self.config['num_self_play_games']} completed")
        
        return training_data
    
    def train_neural_network(self):
        """
        Train neural network on replay buffer data
        """
        if len(self.replay_buffer) < self.config['batch_size']:
            print("Not enough data in replay buffer. Skipping training.")
            return
        
        print(f"Training neural network for {self.config['epochs_per_iteration']} epochs...")
        
        # Prepare dataset
        sample_data = random.sample(self.replay_buffer, min(len(self.replay_buffer), 50000))
        states, policies, values = zip(*sample_data)
        
        dataset = SelfPlayDataset(states, policies, values)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.config['epochs_per_iteration']):
            total_loss = 0
            policy_loss_sum = 0
            value_loss_sum = 0
            
            for states_batch, policies_batch, values_batch in dataloader:
                # Forward pass
                policy_out, value_out = self.model(states_batch)[:2]
                
                # Calculate losses
                policy_loss = self.policy_loss_fn(policy_out, policies_batch)
                value_loss = self.value_loss_fn(value_out.squeeze(), values_batch.squeeze())
                
                # Combined loss
                loss = (self.config['policy_loss_weight'] * policy_loss + 
                        self.config['value_loss_weight'] * value_loss)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Accumulate losses
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
            
            # Print epoch stats
            avg_loss = total_loss / len(dataloader)
            avg_policy_loss = policy_loss_sum / len(dataloader)
            avg_value_loss = value_loss_sum / len(dataloader)
            
            print(f"Epoch {epoch+1}/{self.config['epochs_per_iteration']}, "
                  f"Loss: {avg_loss:.4f}, "
                  f"Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}")
    
    def evaluate(self):
        """
        Evaluate model by playing against a previous version or random agent
        
        Returns:
            Win rate against baseline
        """
        print(f"Evaluating model with {self.config['eval_games']} games...")
        
        # For this simple implementation, evaluate against random policy
        wins = 0
        draws = 0
        losses = 0
        
        for game in range(self.config['eval_games']):
            # Initialize game state
            state = self.game_env.get_initial_state()
            current_player = 0  # Starting player (0 = current model, 1 = random)
            
            # Play game
            while not self.game_env.is_terminal(state):
                if current_player == 0:
                    # Current model plays
                    action_probs = self.mcts.search(state)
                    action = np.argmax(action_probs)  # Deterministic in evaluation
                else:
                    # Random policy plays
                    valid_actions = self.game_env.get_valid_actions(state)
                    valid_indices = np.where(valid_actions == 1)[0]
                    action = np.random.choice(valid_indices)
                
                # Apply action
                state = self.game_env.step(state, action)
                
                # Switch player
                current_player = 1 - current_player
            
            # Game over, determine outcome
            final_reward = self.game_env.get_reward(state)
            
            if final_reward == 1:
                # Current model's perspective
                if current_player == 1:  # Current model made the last move
                    wins += 1
                else:
                    losses += 1
            elif final_reward == -1:
                if current_player == 1:  # Random policy made the last move
                    losses += 1
                else:
                    wins += 1
            else:
                draws += 1
            
            print(f"Evaluation game {game+1}/{self.config['eval_games']} completed")
        
        # Calculate win rate
        win_rate = (wins + 0.5 * draws) / self.config['eval_games']
        
        print(f"Evaluation results: {wins} wins, {draws} draws, {losses} losses")
        print(f"Win rate against random policy: {win_rate:.2f}")
        
        return win_rate
    
    def save_checkpoint(self, iteration, is_best=False):
        """
        Save model checkpoint
        
        Args:
            iteration: Current iteration number
            is_best: Whether this is the best model so far
        """
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f"model_iteration_{iteration}.pt"
        )
        
        if is_best:
            checkpoint_path = os.path.join(
                self.config['checkpoint_dir'],
                "best_model.pt"
            )
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': iteration,
            'config': self.config
        }
        
        if self.typhoon_model:
            checkpoint['typhoon_model'] = self.typhoon_model.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Update config if loaded from checkpoint
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        print(f"Checkpoint loaded from {checkpoint_path}") 