import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from collections import defaultdict


class KnowledgeGraph:
    """
    Knowledge Graph for enhancing MCTS with domain knowledge
    """
    
    def __init__(self, embedding_size=10):
        """
        Initialize Knowledge Graph
        
        Args:
            embedding_size: Size of node embeddings
        """
        # Create NetworkX graph
        self.graph = nx.DiGraph()
        
        # Store node and edge metadata
        self.node_types = {}
        self.edge_types = {}
        
        # Store embeddings
        self.embedding_size = embedding_size
        self.node_embeddings = {}
        
        # Embedding model
        self.embedding_model = KnowledgeGraphEmbedding(embedding_size)
        
    def add_node(self, node_id, node_type, attributes=None):
        """
        Add a node to the knowledge graph
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of the node (e.g., 'state', 'action', 'pattern')
            attributes: Dictionary of node attributes
        """
        # Add node to graph
        self.graph.add_node(node_id, type=node_type, **(attributes or {}))
        
        # Store node type
        self.node_types[node_id] = node_type
        
        # Initialize random embedding for node
        self.node_embeddings[node_id] = np.random.normal(0, 0.1, self.embedding_size)
        
    def add_edge(self, from_node, to_node, edge_type, weight=1.0, attributes=None):
        """
        Add an edge to the knowledge graph
        
        Args:
            from_node: Source node ID
            to_node: Target node ID
            edge_type: Type of the edge (e.g., 'leads_to', 'similar', 'causes')
            weight: Edge weight
            attributes: Dictionary of edge attributes
        """
        # Add edge to graph
        self.graph.add_edge(from_node, to_node, type=edge_type, 
                           weight=weight, **(attributes or {}))
        
        # Store edge type
        self.edge_types[(from_node, to_node)] = edge_type
        
    def get_node_embedding(self, node_id):
        """
        Get embedding for a node
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node embedding as numpy array
        """
        return self.node_embeddings.get(node_id, 
                                       np.zeros(self.embedding_size))
    
    def get_state_embedding(self, state, state_encoder=None):
        """
        Get embedding for a state by combining relevant node embeddings
        
        Args:
            state: Game state
            state_encoder: Optional function to encode state to string
            
        Returns:
            State embedding as numpy array
        """
        # Convert state to string if encoder provided
        if state_encoder:
            state_str = state_encoder(state)
        else:
            state_str = str(state)
        
        # If state exists in graph, use its embedding
        if state_str in self.node_embeddings:
            return self.node_embeddings[state_str]
        
        # Otherwise, find similar states and combine their embeddings
        similar_states = self.find_similar_nodes(state_str, node_type='state')
        
        if not similar_states:
            # No similar states found, return zero embedding
            return np.zeros(self.embedding_size)
        
        # Combine embeddings of similar states
        combined_embedding = np.zeros(self.embedding_size)
        total_weight = 0
        
        for similar_state, similarity in similar_states:
            state_embedding = self.node_embeddings.get(similar_state, 
                                                     np.zeros(self.embedding_size))
            combined_embedding += similarity * state_embedding
            total_weight += similarity
        
        if total_weight > 0:
            combined_embedding /= total_weight
            
        return combined_embedding
    
    def find_similar_nodes(self, node_str, node_type=None, top_k=3):
        """
        Find similar nodes in the graph
        
        Args:
            node_str: Node identifier or representation
            node_type: Optional node type to filter
            top_k: Number of similar nodes to return
            
        Returns:
            List of (node_id, similarity) pairs
        """
        # Simple implementation: can be enhanced with more sophisticated
        # similarity metrics based on graph structure or domain knowledge
        
        similarities = []
        
        for node_id in self.graph.nodes():
            # Skip if node type doesn't match
            if node_type and self.node_types.get(node_id) != node_type:
                continue
                
            # Calculate string similarity (very basic)
            # This should be replaced with more advanced similarity measures
            similarity = self._calculate_string_similarity(node_str, str(node_id))
            
            if similarity > 0.5:  # Threshold
                similarities.append((node_id, similarity))
        
        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _calculate_string_similarity(self, str1, str2):
        """
        Calculate string similarity
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple Jaccard similarity of character sets
        # Can be replaced with more advanced metrics
        set1 = set(str1)
        set2 = set(str2)
        
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def learn_embeddings(self, data, epochs=100, batch_size=32, learning_rate=0.01):
        """
        Learn node embeddings from data
        
        Args:
            data: List of (source, relation, target) triples
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        # Create optimizer
        optimizer = torch.optim.Adam(self.embedding_model.parameters(), lr=learning_rate)
        
        # Convert node IDs to indices
        node_to_idx = {node: i for i, node in enumerate(self.graph.nodes())}
        relation_to_idx = {rel: i for i, rel in enumerate(set(self.edge_types.values()))}
        
        # Convert data to tensors
        sources = [node_to_idx[s] for s, _, _ in data]
        relations = [relation_to_idx[r] for _, r, _ in data]
        targets = [node_to_idx[t] for _, _, t in data]
        
        sources = torch.tensor(sources)
        relations = torch.tensor(relations)
        targets = torch.tensor(targets)
        
        # Initialize embeddings matrix
        embeddings = np.zeros((len(node_to_idx), self.embedding_size))
        for node, idx in node_to_idx.items():
            embeddings[idx] = self.node_embeddings.get(node, 
                                                     np.random.normal(0, 0.1, self.embedding_size))
        
        self.embedding_model.init_embeddings(torch.FloatTensor(embeddings), len(relation_to_idx))
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(data))
            sources_shuffled = sources[indices]
            relations_shuffled = relations[indices]
            targets_shuffled = targets[indices]
            
            # Mini-batch training
            total_loss = 0
            num_batches = (len(data) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, len(data))
                
                s_batch = sources_shuffled[start:end]
                r_batch = relations_shuffled[start:end]
                t_batch = targets_shuffled[start:end]
                
                # Generate negative samples
                t_neg_batch = torch.randint(0, len(node_to_idx), (end - start,))
                
                # Forward pass
                pos_scores = self.embedding_model(s_batch, r_batch, t_batch)
                neg_scores = self.embedding_model(s_batch, r_batch, t_neg_batch)
                
                # Compute loss
                loss = self.embedding_model.margin_loss(pos_scores, neg_scores)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}")
        
        # Update node embeddings
        node_embeddings = self.embedding_model.entity_embeddings.weight.detach().numpy()
        for node, idx in node_to_idx.items():
            self.node_embeddings[node] = node_embeddings[idx]
    
    def save(self, filename):
        """
        Save the knowledge graph and embeddings
        
        Args:
            filename: File to save to
        """
        data = {
            'node_types': self.node_types,
            'edge_types': self.edge_types,
            'embeddings': self.node_embeddings,
            'graph': nx.to_dict_of_dicts(self.graph)
        }
        np.save(filename, data, allow_pickle=True)
    
    def load(self, filename):
        """
        Load the knowledge graph and embeddings
        
        Args:
            filename: File to load from
        """
        data = np.load(filename, allow_pickle=True).item()
        
        self.node_types = data['node_types']
        self.edge_types = data['edge_types']
        self.node_embeddings = data['embeddings']
        self.graph = nx.from_dict_of_dicts(data['graph'], create_using=nx.DiGraph())


class KnowledgeGraphEmbedding(nn.Module):
    """
    Neural network for learning knowledge graph embeddings
    """
    
    def __init__(self, embedding_size, margin=1.0):
        """
        Initialize embedding model
        
        Args:
            embedding_size: Size of embeddings
            margin: Margin for loss function
        """
        super(KnowledgeGraphEmbedding, self).__init__()
        
        self.embedding_size = embedding_size
        self.margin = margin
        
    def init_embeddings(self, entity_embeddings, num_relations):
        """
        Initialize embeddings
        
        Args:
            entity_embeddings: Initial entity embeddings
            num_relations: Number of relation types
        """
        num_entities = entity_embeddings.shape[0]
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, self.embedding_size)
        self.entity_embeddings.weight.data.copy_(entity_embeddings)
        
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, self.embedding_size)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
    def forward(self, head_indices, relation_indices, tail_indices):
        """
        Forward pass
        
        Args:
            head_indices: Indices of head entities
            relation_indices: Indices of relations
            tail_indices: Indices of tail entities
            
        Returns:
            Scores for triples
        """
        # Get embeddings
        head_embeddings = self.entity_embeddings(head_indices)
        relation_embeddings = self.relation_embeddings(relation_indices)
        tail_embeddings = self.entity_embeddings(tail_indices)
        
        # Calculate scores (TransE: head + relation ≈ tail)
        scores = head_embeddings + relation_embeddings - tail_embeddings
        scores = -torch.norm(scores, p=2, dim=1)
        
        return scores
    
    def margin_loss(self, pos_scores, neg_scores):
        """
        Margin-based ranking loss
        
        Args:
            pos_scores: Scores for positive triples
            neg_scores: Scores for negative triples
            
        Returns:
            Loss value
        """
        return torch.mean(torch.relu(self.margin - pos_scores + neg_scores)) 