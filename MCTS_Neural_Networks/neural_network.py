import torch
import torch.nn as nn
import torch.nn.functional as F


class MCTSNet(nn.Module):
    """
    Neural Network for MCTS that outputs policy and value
    """
    def __init__(self, input_size, action_size, hidden_size=256, knowledge_size=0):
        super(MCTSNet, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.knowledge_size = knowledge_size
        
        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_size + knowledge_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Policy head
        self.policy_head = nn.Linear(hidden_size, action_size)
        
        # Value head
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Temporal value head for predicting future value
        self.temporal_value_head = nn.Linear(hidden_size, 1)
        
        # Confidence head
        self.confidence_head = nn.Linear(hidden_size, 1)
        
        # Difficulty prediction head (easy, medium, hard)
        self.difficulty_head = nn.Linear(hidden_size, 3)
        
        # Time embedding for temporal reasoning
        self.time_embedding = nn.Embedding(100, hidden_size)
        
    def forward(self, x, knowledge_embedding=None, time_step=0):
        """
        Forward pass
        
        Args:
            x: State representation
            knowledge_embedding: Optional knowledge graph embedding
            time_step: Time step for temporal reasoning
            
        Returns:
            policy: Action probabilities
            value: State value
            temporal_value: Predicted future value
            confidence: Confidence in prediction
            difficulty: Predicted difficulty level
        """
        batch_size = x.size(0)
        
        # Concatenate knowledge embedding if provided
        if knowledge_embedding is not None and self.knowledge_size > 0:
            x = torch.cat([x, knowledge_embedding], dim=1)
        elif self.knowledge_size > 0:
            # If knowledge size is specified but no embedding provided, use zeros
            zero_knowledge = torch.zeros(batch_size, self.knowledge_size, device=x.device)
            x = torch.cat([x, zero_knowledge], dim=1)
        
        # Get backbone features
        features = self.backbone(x)
        
        # Add time embedding for temporal reasoning if time_step > 0
        if time_step > 0:
            time_embed = self.time_embedding(torch.tensor([min(time_step, 99)], 
                                                         device=x.device))
            # Add time embedding to features
            features = features + time_embed.expand_as(features)
        
        # Policy head
        policy_logits = self.policy_head(features)
        policy = F.softmax(policy_logits, dim=1)
        
        # Value head
        value = torch.tanh(self.value_head(features))
        
        # Temporal value head
        temporal_value = torch.tanh(self.temporal_value_head(features))
        
        # Confidence head
        confidence = torch.sigmoid(self.confidence_head(features))
        
        # Difficulty head
        difficulty = F.softmax(self.difficulty_head(features), dim=1)
        
        return policy, value, temporal_value, confidence, difficulty


class EnhancedMCTSNet(nn.Module):
    """
    Enhanced Neural Network for MCTS with parallel processing
    using residual connections and attention mechanisms
    """
    def __init__(self, input_size, action_size, hidden_size=256, knowledge_size=0, 
                 num_residual_blocks=3, num_heads=4):
        super(EnhancedMCTSNet, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.knowledge_size = knowledge_size
        
        # Initial processing
        self.input_layer = nn.Sequential(
            nn.Linear(input_size + knowledge_size, hidden_size),
            nn.ReLU()
        )
        
        # Residual blocks with attention
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, num_heads) 
            for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Temporal value head
        self.temporal_value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Difficulty head
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3)
        )
        
        # Time embedding
        self.time_embedding = nn.Embedding(100, hidden_size)
        
    def forward(self, x, knowledge_embedding=None, time_step=0):
        batch_size = x.size(0)
        
        # Concatenate knowledge embedding if provided
        if knowledge_embedding is not None and self.knowledge_size > 0:
            x = torch.cat([x, knowledge_embedding], dim=1)
        elif self.knowledge_size > 0:
            # If knowledge size is specified but no embedding provided, use zeros
            zero_knowledge = torch.zeros(batch_size, self.knowledge_size, device=x.device)
            x = torch.cat([x, zero_knowledge], dim=1)
        
        # Initial processing
        x = self.input_layer(x)
        
        # Add time embedding
        if time_step > 0:
            time_embed = self.time_embedding(torch.tensor([min(time_step, 99)], 
                                                        device=x.device))
            x = x + time_embed.expand_as(x)
        
        # Process through residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy output
        policy_logits = self.policy_head(x)
        policy = F.softmax(policy_logits, dim=1)
        
        # Value output
        value = torch.tanh(self.value_head(x))
        
        # Temporal value output
        temporal_value = torch.tanh(self.temporal_value_head(x))
        
        # Confidence output
        confidence = torch.sigmoid(self.confidence_head(x))
        
        # Difficulty output
        difficulty = F.softmax(self.difficulty_head(x), dim=1)
        
        return policy, value, temporal_value, confidence, difficulty


class ResidualBlock(nn.Module):
    """
    Residual block with attention mechanism
    """
    def __init__(self, hidden_size, num_heads):
        super(ResidualBlock, self).__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_input = x.unsqueeze(1)  # Add sequence dimension
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        x = x + attn_output.squeeze(1)  # Residual connection
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + ffn_output  # Residual connection
        x = self.norm2(x)
        
        return x 