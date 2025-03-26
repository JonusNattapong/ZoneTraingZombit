import argparse
import torch
import os
import numpy as np
import random
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_network import MCTSNet, EnhancedMCTSNet
from datasets import load_dataset
from game_environment import GameEnvironment
from mcts import MCTS
from parallel_mcts import ParallelMCTS
from trainer import SelfPlayTrainer
from knowledge_graph import KnowledgeGraph


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train and play with MCTS + Neural Networks')
    parser.add_argument('--mode', type=str, choices=['train', 'play', 'evaluate'], 
                       default='train', help='Mode: train, play, or evaluate')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='Path to model checkpoint for play/evaluate mode')
    parser.add_argument('--enhanced', action='store_true', 
                       help='Use enhanced neural network architecture')
    parser.add_argument('--parallel', action='store_true', 
                       help='Use parallel MCTS for self-play')
    parser.add_argument('--num_iterations', type=int, default=100, 
                       help='Number of training iterations')
    parser.add_argument('--hidden_size', type=int, default=256, 
                       help='Hidden layer size for neural network')
    parser.add_argument('--batch_size', type=int, default=256, 
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--use_knowledge', action='store_true', 
                       help='Use knowledge graph')
    parser.add_argument('--knowledge_size', type=int, default=10, 
                       help='Knowledge embedding size')
    parser.add_argument('--num_self_play', type=int, default=100, 
                       help='Number of self-play games per iteration')
    parser.add_argument('--mcts_sims', type=int, default=800, 
                       help='Number of MCTS simulations per move')
    parser.add_argument('--num_processes', type=int, default=4, 
                       help='Number of processes for parallel MCTS')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(args, input_size, action_size):
    """Create neural network model"""
    if args.enhanced:
        return EnhancedMCTSNet(
            input_size=input_size,
            action_size=action_size,
            hidden_size=args.hidden_size,
            knowledge_size=args.knowledge_size if args.use_knowledge else 0
        )
    else:
        return MCTSNet(
            input_size=input_size,
            action_size=action_size,
            hidden_size=args.hidden_size,
            knowledge_size=args.knowledge_size if args.use_knowledge else 0
        )

def load_model(model, checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


class DatasetEnvironment(GameEnvironment):
    """Game environment that uses a dataset for text generation MCTS"""
    
    def __init__(self, dataset_path="datasets/dataset_1000.json"):
        """Initialize with dataset and tracking variables"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
        except FileNotFoundError:
            print(f"Dataset file not found: {dataset_path}")
            print("Creating minimal dummy dataset for testing")
            self.dataset = [
                {"prompt": "Explain Monte Carlo Tree Search", 
                 "completion": "Monte Carlo Tree Search (MCTS) is an algorithm that combines tree search with random sampling for decision making."},
                {"prompt": "What is reinforcement learning?", 
                 "completion": "Reinforcement learning is a type of machine learning where agents learn to make decisions by taking actions in an environment."}
            ]
        except json.JSONDecodeError:
            print(f"Error parsing JSON from: {dataset_path}")
            self.dataset = []
            
        self.current_idx = 0
        self.action_space = 10  # Default action space size
        self.current_text = ""  # Track currently generated text
        self.max_tokens = 20    # Maximum tokens to generate per sample
        self.token_options = []  # Store possible next tokens
        self.current_step = 0   # Track steps within current sample
        self.vocabulary = self._build_vocabulary()
        self.embedding_cache = {}  # Cache for word embeddings
        self.vector_size = 100  # Size of state encoding vector
        
    def _build_vocabulary(self):
        """Build vocabulary from dataset for better tokenization"""
        vocabulary = set()
        # Extract all words from prompts and completions
        for item in self.dataset:
            prompt = item.get("prompt", "")
            completion = item.get("completion", "")
            
            words = (prompt + " " + completion).lower().split()
            vocabulary.update(words)
            
        # Add special tokens
        vocabulary.update(["[START]", "[END]", "[UNK]"])
        
        # Convert to dictionary for faster lookup
        return {word: idx for idx, word in enumerate(vocabulary)}
        
    def get_initial_state(self):
        """Return initial state with the prompt from current dataset item"""
        if self.current_idx >= len(self.dataset):
            self.current_idx = 0  # Reset to beginning of dataset
            
        sample = self.dataset[self.current_idx].copy()  # Make a copy to avoid modifying original
        self.current_text = ""  # Reset the generated text
        self.current_step = 0   # Reset step counter
        
        # Generate token options for the given prompt
        completion = sample.get("completion", "")
        if completion:
            # Extract token options from completion
            self.token_options = self._extract_token_options(completion)
            self.action_space = min(len(self.token_options), 10)  # Limit to reasonable size
        
        # Add history tracking
        sample['generated_text'] = ""
        sample['history'] = []
        sample['step'] = 0
        
        return sample
        
    def get_valid_actions(self, state):
        """Return valid actions based on current state in dataset"""
        # Create actions array based on current action space
        valid_actions = np.ones(self.action_space_size())
        
        # If we've already generated max tokens, limit actions
        if len(self.current_text.split()) >= self.max_tokens:
            valid_actions = np.zeros(self.action_space_size())
            valid_actions[0] = 1  # Only allow ending the generation
            return valid_actions
        
        # If we have specific token options, implement more sophisticated validation
        if len(self.token_options) > 0:
            # Get current context
            context = state.get('generated_text', '')
            
            # If using a language model, we could check probabilities here
            # Simple grammar rules (for demonstration):
            if context and len(context.split()) > 0:
                last_token = context.split()[-1].lower()
                
                # Example rules (these would be much more sophisticated in a real implementation)
                if last_token.endswith('.'):
                    # After a period, capitalize the next word
                    for i, token in enumerate(self.token_options):
                        if i < len(valid_actions) and not token.capitalize() == token:
                            valid_actions[i] *= 0.8  # Reduce probability but don't eliminate
                            
                if last_token in ['a', 'an', 'the']:
                    # After articles, next token is likely a noun or adjective
                    # This is oversimplified, but demonstrates the concept
                    for i, token in enumerate(self.token_options):
                        if i < len(valid_actions) and token in ['and', 'or', 'but', 'if', 'then']:
                            valid_actions[i] *= 0.5  # Reduce probability for unlikely transitions
        
        # Ensure at least one action is valid
        if np.sum(valid_actions) == 0:
            valid_actions[0] = 1
            
        return valid_actions
        
    def step(self, state, action):
        """Apply action to current state and return new state"""
        # Get the token corresponding to this action
        selected_token = ""
        if action < len(self.token_options):
            selected_token = self.token_options[action]
        else:
            # Fallback for when action doesn't map to a token
            selected_token = f"[TOKEN_{action}]"
            
        # Append token to current text with proper spacing
        if self.current_text and not (self.current_text.endswith(' ') or selected_token.startswith(' ')):
            self.current_text += " " + selected_token
        else:
            self.current_text += selected_token
            
        # Clean up any double spaces
        self.current_text = ' '.join(self.current_text.split())
        
        # Update state with the generated text
        new_state = state.copy()
        new_state['generated_text'] = self.current_text
        new_state['step'] = self.current_step + 1
        
        # Track history of actions and tokens
        if 'history' not in new_state:
            new_state['history'] = []
        new_state['history'].append({
            'action': action,
            'token': selected_token,
            'step': self.current_step
        })
        
        # Update step counter
        self.current_step += 1
        
        # Generate new token options based on updated context
        if self.current_idx < len(self.dataset):
            current_sample = self.dataset[self.current_idx]
            completion = current_sample.get("completion", "")
            
            # Use n-gram model to predict next options
            next_options = self._predict_next_tokens(self.current_text, completion)
            self.token_options = next_options
            self.action_space = min(len(self.token_options), 10)
            
        # Move to next sample if we've reached max tokens or special end token
        if len(self.current_text.split()) >= self.max_tokens or selected_token == "[END]" or action == 0:
            self.current_idx += 1
            
            # Reset for next sample
            if self.current_idx < len(self.dataset):
                next_sample = self.dataset[self.current_idx]
                completion = next_sample.get("completion", "")
                
                # Reset tracking variables
                self.current_text = ""
                self.current_step = 0
                
                # Generate options for next sample
                if completion:
                    self.token_options = self._extract_token_options(completion)
                    self.action_space = min(len(self.token_options), 10)
        
        return new_state
        
    def _predict_next_tokens(self, context, target, n=10):
        """Predict next possible tokens based on context and target"""
        if not context or not target:
            return self._extract_token_options(target)
            
        # Simple n-gram based prediction
        context_words = context.lower().split()
        target_words = target.lower().split()
        
        # Find position in target text similar to our context
        match_position = -1
        context_len = len(context_words)
        
        if context_len > 0:
            # Look for the closest match to our generated context
            for i in range(len(target_words) - context_len + 1):
                matching = True
                for j in range(min(3, context_len)):  # Check last few words
                    if i + context_len - j - 1 >= len(target_words):
                        matching = False
                        break
                    if context_len - j - 1 < 0:
                        break
                    if context_words[context_len - j - 1] != target_words[i + context_len - j - 1]:
                        matching = False
                        break
                
                if matching:
                    match_position = i + context_len
                    break
        
        # Get next tokens based on match position
        next_tokens = []
        if match_position >= 0 and match_position < len(target_words):
            # Add the actual next words from target
            next_tokens = target_words[match_position:match_position + n]
        else:
            # If no match found, extract tokens from target
            next_tokens = self._extract_token_options(target, max_options=n)
            
        # Add some random words from vocabulary for exploration
        if len(next_tokens) < n:
            voc_items = list(self.vocabulary.keys())
            random_tokens = random.sample(voc_items, min(n - len(next_tokens), len(voc_items)))
            next_tokens.extend(random_tokens)
            
        # Ensure we have exactly n tokens
        if len(next_tokens) > n:
            next_tokens = next_tokens[:n]
        while len(next_tokens) < n:
            next_tokens.append("[UNK]")
            
        return next_tokens
        
    def is_terminal(self, state):
        """Check if current state is terminal"""
        # Terminal conditions:
        # 1. End of dataset
        # 2. Max tokens generated
        # 3. Special end token generated
        # 4. Max steps reached
        
        if self.current_idx >= len(self.dataset):
            return True
            
        generated_text = state.get('generated_text', "")
        if generated_text and len(generated_text.split()) >= self.max_tokens:
            return True
            
        if "[END]" in generated_text:
            return True
            
        if state.get('step', 0) >= self.max_tokens * 2:  # Safety limit
            return True
            
        return False
        
    def get_reward(self, state):
        """Calculate reward based on similarity to expected completion"""
        if self.current_idx > len(self.dataset):
            return 0  # End of dataset, neutral reward
            
        # Get current dataset sample
        sample_idx = min(max(0, self.current_idx - 1), len(self.dataset) - 1)
        sample = self.dataset[sample_idx]
        expected_completion = sample.get("completion", "")
        generated_text = state.get('generated_text', "")
        
        # Empty states get neutral reward
        if not generated_text or not expected_completion:
            return 0
            
        # Calculate multiple similarity metrics for better reward
        rewards = []
        
        # 1. Token overlap (Jaccard similarity)
        expected_tokens = set(expected_completion.lower().split())
        generated_tokens = set(generated_text.lower().split())
        
        if expected_tokens and generated_tokens:
            intersection = len(expected_tokens.intersection(generated_tokens))
            union = len(expected_tokens.union(generated_tokens))
            jaccard = intersection / max(1, union)
            rewards.append(jaccard)
        
        # 2. N-gram similarity (bigrams and trigrams)
        expected_bigrams = self._get_ngrams(expected_completion, 2)
        generated_bigrams = self._get_ngrams(generated_text, 2)
        
        if expected_bigrams and generated_bigrams:
            intersection = len(expected_bigrams.intersection(generated_bigrams))
            union = len(expected_bigrams.union(generated_bigrams))
            bigram_sim = intersection / max(1, union)
            rewards.append(bigram_sim)
        
        # 3. Length penalty - penalize if too short or too long
        expected_len = len(expected_completion.split())
        generated_len = len(generated_text.split())
        length_diff = abs(expected_len - generated_len) / max(expected_len, 1)
        length_penalty = max(0, 1 - length_diff)  # 1 if perfect length, decreasing as difference increases
        rewards.append(length_penalty)
        
        # 4. Sequence match - reward for correct sequence of tokens
        seq_match = self._longest_common_subsequence(
            expected_completion.lower().split(),
            generated_text.lower().split()
        )
        seq_match_ratio = seq_match / max(len(expected_completion.split()), 1)
        rewards.append(seq_match_ratio)
        
        # Combine rewards with different weights
        weights = [0.4, 0.3, 0.1, 0.2]  # Adjust weights based on importance
        combined_reward = sum(r * w for r, w in zip(rewards, weights[:len(rewards)]))
        
        # Scale to range [-1, 1] to match MCTS assumptions
        scaled_reward = (combined_reward * 2) - 1
        
        # Add step-based reward decay to encourage efficient generation
        step_penalty = min(1.0, state.get('step', 0) / (self.max_tokens * 2))
        final_reward = scaled_reward * (1 - step_penalty * 0.5)
        
        return final_reward
        
    def _get_ngrams(self, text, n):
        """Extract n-grams from text"""
        tokens = text.lower().split()
        ngrams = set()
        
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.add(ngram)
            
        return ngrams
        
    def _longest_common_subsequence(self, seq1, seq2):
        """Find length of longest common subsequence between two sequences"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        return dp[m][n]
        
    def get_encoded_state(self, state):
        """Encode state for neural network input using better vectorization"""
        # Initialize encoding vector
        encoding = np.zeros(self.vector_size) 
        
        if not isinstance(state, dict):
            return encoding
            
        # Extract features
        prompt = state.get('prompt', "")
        generated = state.get('generated_text', "")
        step = state.get('step', 0)
        
        # Combine text for encoding
        combined_text = prompt + " " + generated
        
        # TF-IDF inspired encoding
        words = combined_text.lower().split()
        word_counts = {}
        
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
                
        # Calculate "TF" component
        for word, count in word_counts.items():
            # Get word index from vocabulary or hash it
            if word in self.vocabulary:
                idx = self.vocabulary[word] % self.vector_size
            else:
                idx = hash(word) % self.vector_size
                
            # TF component: word frequency in document
            tf = count / max(1, len(words))
            
            # Add to encoding with position-based weighting
            # Words at the end of generated text get higher weight
            position_boost = 1.0
            if word in generated.split()[-3:]:
                position_boost = 2.0  # Boost recent words
                
            encoding[idx] += tf * position_boost
            
        # Add step information (normalized)
        step_idx = self.vector_size - 1
        encoding[step_idx] = min(1.0, step / self.max_tokens)
        
        # Normalize vector
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding = encoding / norm
            
        return encoding
        
    def action_space_size(self):
        """Return the number of possible actions"""
        return max(10, self.action_space)  # Ensure minimum action space size
        
    def get_current_player(self, state):
        """Always return 0 as this is a single-player environment"""
        return 0
        
    def _extract_token_options(self, text, max_options=10):
        """Extract possible token options from text with improved semantics"""
        if not text:
            return ["[UNK]"] * max_options
            
        # Tokenize by splitting on spaces
        words = text.split()
        
        if len(words) <= max_options:
            # Pad with special tokens if needed
            options = words.copy()
            while len(options) < max_options:
                options.append("[UNK]")
            return options
        
        # Strategy: Get tokens from different parts of the text
        options = []
        
        # Take tokens from beginning
        options.extend(words[:max_options // 3])
        
        # Take tokens from middle
        mid_start = len(words) // 2 - (max_options // 6)
        mid_end = min(len(words), mid_start + (max_options // 3))
        options.extend(words[mid_start:mid_end])
        
        # Take tokens from end
        end_start = max(0, len(words) - (max_options // 3))
        options.extend(words[end_start:])
        
        # Add special tokens
        if len(options) < max_options:
            options.append("[END]")
            
        # Ensure we have exactly max_options
        if len(options) > max_options:
            options = options[:max_options]
        while len(options) < max_options:
            options.append("[UNK]")
            
        return options
        
    def render(self, state):
        """Display the current state with enhanced formatting"""
        if not isinstance(state, dict):
            print("Invalid state format")
            return
            
        print("\n" + "=" * 60)
        prompt = state.get('prompt', "")
        generated = state.get('generated_text', "")
        step = state.get('step', 0)
        
        # Show limited prompt for readability
        if len(prompt) > 100:
            print(f"PROMPT: {prompt[:97]}...")
        else:
            print(f"PROMPT: {prompt}")
            
        # Show generated text with highlighting
        print(f"\nGENERATED TEXT (step {step}):")
        print(f">> {generated}")
        
        # Show expected output
        if self.current_idx < len(self.dataset):
            expected = self.dataset[self.current_idx].get('completion', "")
            if len(expected) > 100:
                print(f"\nEXPECTED: {expected[:97]}...")
            else:
                print(f"\nEXPECTED: {expected}")
                
        # Show available actions
        print("\nAVAILABLE ACTIONS:")
        for i, token in enumerate(self.token_options[:min(5, len(self.token_options))]):
            print(f"  {i}: '{token}'")
        if len(self.token_options) > 5:
            print(f"  ...and {len(self.token_options) - 5} more options")
            
        # Show current reward
        reward = self.get_reward(state)
        print(f"\nCURRENT REWARD: {reward:.3f}")
        print("=" * 60)


def load_typhoon_model(model_path="checkpoints/typhoon-7b"):
    """Load the typhoon-7b model from local path"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Typhoon model not found at {model_path}")
    
    print(f"Loading typhoon-7b model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer



def play_game(model, game_env, num_simulations=800, human_player=0):
    """
    Play a game against the model
    
    Args:
        model: Neural network model
        game_env: Game environment
        num_simulations: Number of MCTS simulations per move
        human_player: Player index for human (0 = first player, 1 = second player)
    """
    # Create MCTS
    mcts = MCTS(
        model=model,
        game_env=game_env,
        num_simulations=num_simulations,
        dirichlet_noise=False,  # No exploration noise in human games
        temperature=0.1  # Low temperature for deterministic play
    )
    
    # Initialize game
    state = game_env.get_initial_state()
    game_env.render(state)
    
    # Play game
    while not game_env.is_terminal(state):
        current_player = game_env.get_current_player(state)
        
        if current_player == human_player:
            # Human player's turn
            while True:
                try:
                    print("Enter action (0-9):")
                    action = int(input().strip())
                    
                    # Check if action is valid
                    valid_actions = game_env.get_valid_actions(state)
                    if action >= 0 and action < len(valid_actions) and valid_actions[action] == 1:
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Try again.")
        else:
            # AI player's turn
            print("AI thinking...")
            action_probs = mcts.search(state)
            action = np.argmax(action_probs)
            print(f"AI chose move: {action}")
        
        # Apply action
        state = game_env.step(state, action)
        game_env.render(state)
    
    # Game over
    reward = game_env.get_reward(state)
    
    if reward == 0:
        print("Game ended in a draw!")
    elif (reward == 1 and human_player == 0) or (reward == -1 and human_player == 1):
        print("You won!")
    else:
        print("AI won!")


def main():
    """Main function"""
    args = parse_args()
    set_seed(args.seed)
    
    # Create dataset environment
    game_env = DatasetEnvironment()
    
    # Determine input and action sizes
    state = game_env.get_initial_state()
    encoded_state = game_env.get_encoded_state(state)
    input_size = len(encoded_state)
    action_size = game_env.action_space_size()
    
    # Create or load models
    model = create_model(args, input_size, action_size)
    if args.model_path and os.path.exists(args.model_path):
        model = load_model(model, args.model_path)
        print(f"Loaded MCTS model from {args.model_path}")
    else:
        print("Created new MCTS model")
    
    # Load typhoon-7b model
    try:
        typhoon_model, typhoon_tokenizer = load_typhoon_model()
        print("Successfully loaded typhoon-7b model")
    except Exception as e:
        print(f"Error loading typhoon-7b model: {e}")
        typhoon_model = None
    
    # Create knowledge graph if needed
    knowledge_graph = None
    if args.use_knowledge:
        knowledge_graph = KnowledgeGraph(embedding_size=args.knowledge_size)
        # Here you would initialize the knowledge graph with domain knowledge
        # This is just a placeholder
        print("Created knowledge graph")
    
    # Different modes
    if args.mode == 'train':
        # Configure training
        config = {
            'num_iterations': args.num_iterations,
            'num_self_play_games': args.num_self_play,
            'num_mcts_simulations': args.mcts_sims,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'use_parallel_mcts': args.parallel,
            'num_processes': args.num_processes
        }
        
        # Create trainer and train
        trainer = SelfPlayTrainer(model, game_env, config, typhoon_model=typhoon_model)
        trainer.train()
        
    elif args.mode == 'play':
        # Play against the model
        play_game(model, game_env, num_simulations=args.mcts_sims)
        
    elif args.mode == 'evaluate':
        # Evaluate the model
        if args.model_path:
            # Configure training (for evaluation only)
            config = {
                'num_iterations': 1,
                'num_self_play_games': 1,
                'num_mcts_simulations': args.mcts_sims,
                'eval_games': 100,  # More evaluation games
                'use_parallel_mcts': args.parallel,
                'num_processes': args.num_processes
            }
            
            # Create trainer and evaluate
            trainer = SelfPlayTrainer(model, game_env, config)
            trainer.evaluate()
        else:
            print("Model path must be provided for evaluation")


if __name__ == "__main__":
    main()