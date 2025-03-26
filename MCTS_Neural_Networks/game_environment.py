from abc import ABC, abstractmethod
import numpy as np

class GameEnvironment(ABC):
    """Abstract base class for game environments"""
    
    @abstractmethod
    def get_initial_state(self):
        """Return the initial game state"""
        pass
        
    @abstractmethod 
    def get_valid_actions(self, state):
        """Return valid actions for current state"""
        pass
        
    @abstractmethod
    def step(self, state, action):
        """Apply action and return new state"""
        pass
        
    @abstractmethod
    def is_terminal(self, state):
        """Check if state is terminal"""
        pass
        
    @abstractmethod
    def get_reward(self, state):
        """Get reward for current state"""
        pass
        
    @abstractmethod
    def get_encoded_state(self, state):
        """Encode state for neural network input"""
        pass
        
    @abstractmethod
    def action_space_size(self):
        """Return size of action space"""
        pass
        
    @abstractmethod
    def get_current_player(self, state):
        """Get current player index"""
        pass
        
    def get_next_player(self, state):
        """Default implementation for alternating players"""
        return 1 - self.get_current_player(state)
        
    def clone_state(self, state):
        """Default state cloning"""
        return np.copy(state)
        
    def get_state_str(self, state):
        """Default string representation"""
        return str(state)
        
    def render(self, state):
        """Default rendering (no-op)"""
        pass