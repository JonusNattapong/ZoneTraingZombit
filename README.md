# ZoneTraingZombit - MCTS with Neural Networks

A sophisticated implementation of Monte Carlo Tree Search (MCTS) enhanced with neural networks for advanced text generation, reasoning, and deep search capabilities.

## Overview

ZoneTraingZombit combines the power of Monte Carlo Tree Search with neural networks to create a system capable of:

- Deep search and reasoning over text generation tasks
- Multi-level search for varying complexity problems
- Temporal reasoning for better planning
- Knowledge graph integration for deeper context understanding
- Adaptive thought mechanisms that adjust exploration based on problem difficulty
- Parallel processing for improved performance

This implementation is particularly effective for training on Thai and English text datasets, with support for both simple and complex reasoning tasks.

## Features

- **Enhanced Neural Network Architecture**: Residual connections and attention mechanisms
- **Parallel MCTS**: Multi-process search for faster decision making
- **Knowledge Graph Integration**: Semantic understanding and relationship modeling
- **Self-Play Training**: Automatic improvement through reinforcement learning
- **Temporal Reasoning**: Decision making with time as a consideration
- **Adaptive Exploration**: Variable exploration rates based on confidence and difficulty
- **Multi-metric Reward System**: Sophisticated reward calculation for better quality outputs

## Requirements

- Python 3.7+
- PyTorch 1.8.0+
- Transformers 4.20.0+
- NetworkX 2.5.0+
- CUDA-capable GPU (recommended for training)
- Additional packages listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ZoneTraingZombit.git
cd ZoneTraingZombit

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r MCTS_Neural_Networks/requirements.txt

# Download pre-trained models (optional)
python MCTS_Neural_Networks/download_model.py
```

## Environment Configuration

Create a `.env` file in the project root with these variables:

```ini
# Required - Mistral API Key
MISTRAL_API_KEY=your_api_key_here

# Optional Configuration
MISTRAL_BASE_URL=https://api.mistral.ai/v1
MAX_WORKERS=4
REQUESTS_PER_MINUTE=30
MISTRAL_MODEL=mistral-tiny
REQUEST_TIMEOUT=30
TEMPERATURE=0.7
MAX_TOKENS=100
TOP_P=1.0
```

## Dataset Generation

To generate a training dataset:
```bash
python MCTS_Neural_Networks/generate_dataset.py --size 50 --min-length 100
```

Options:
- `--size`: Number of samples to generate (default: 20)
- `--min-length`: Minimum response length in characters (default: 50)
- `--output-dir`: Output directory (default: "datasets")

## Usage

### Training

Train a new model or continue training an existing one:

```bash
# Train with basic configuration
python MCTS_Neural_Networks/main.py --mode train

# Train with enhanced neural network architecture
python MCTS_Neural_Networks/main.py --mode train --enhanced

# Train with parallel MCTS for faster processing
python MCTS_Neural_Networks/main.py --mode train --enhanced --parallel --num_processes 8

# Train with knowledge graph integration
python MCTS_Neural_Networks/main.py --mode train --enhanced --use_knowledge --knowledge_size 20
```

### Playing

Interact with a trained model:

```bash
# Play with default settings
python MCTS_Neural_Networks/main.py --mode play --model_path checkpoints/best_model.pt

# Play with more simulations for deeper thinking
python MCTS_Neural_Networks/main.py --mode play --model_path checkpoints/best_model.pt --mcts_sims 1600
```

### Evaluation

Evaluate a trained model's performance:

```bash
# Basic evaluation
python MCTS_Neural_Networks/main.py --mode evaluate --model_path checkpoints/best_model.pt

# More thorough evaluation with parallel processing
python MCTS_Neural_Networks/main.py --mode evaluate --model_path checkpoints/best_model.pt --parallel --num_processes 4
```

## Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Mode: train, play, or evaluate | train |
| `--model_path` | Path to model checkpoint | None |
| `--enhanced` | Use enhanced neural network architecture | False |
| `--parallel` | Use parallel MCTS for self-play | False |
| `--num_iterations` | Number of training iterations | 100 |
| `--hidden_size` | Hidden layer size for neural network | 256 |
| `--batch_size` | Training batch size | 256 |
| `--learning_rate` | Learning rate | 0.001 |
| `--use_knowledge` | Use knowledge graph | False |
| `--knowledge_size` | Knowledge embedding size | 10 |
| `--num_self_play` | Number of self-play games per iteration | 100 |
| `--mcts_sims` | Number of MCTS simulations per move | 800 |
| `--num_processes` | Number of processes for parallel MCTS | 4 |
| `--seed` | Random seed | 42 |

## Architecture Overview

The system combines multiple components:

1. **MCTS Algorithm**: Core search algorithm with neural guidance
2. **Neural Networks**: Policy and value prediction
3. **Knowledge Graph**: Semantic understanding and relationships
4. **Dataset Environment**: Text-based environment for training and inference
5. **Self-Play Trainer**: Reinforcement learning through self-play

## Text Generation Process

1. **Input**: System receives a prompt
2. **Search**: MCTS explores possible token sequences
3. **Evaluation**: Neural network evaluates states and suggests actions
4. **Selection**: Actions are selected based on MCTS statistics
5. **Output**: Generated text is produced incrementally

## Project Structure

```
ZoneTraingZombit/
├── MCTS_Neural_Networks/
│   ├── main.py                  # Main entry point
│   ├── mcts.py                  # MCTS algorithm 
│   ├── parallel_mcts.py         # Parallel MCTS implementation
│   ├── neural_network.py        # Neural network models
│   ├── knowledge_graph.py       # Knowledge graph implementation
│   ├── game_environment.py      # Abstract environment class
│   ├── trainer.py               # Self-play training system
│   ├── generate_dataset.py      # Dataset generation
│   ├── download_model.py        # Model downloading utility
│   └── requirements.txt         # Project dependencies
├── datasets/                    # Training datasets
├── checkpoints/                 # Saved models
└── .env                         # Environment configuration
```

## Example Output

When running in play mode, you'll see output like this:

```
============================================================
PROMPT: Explain Monte Carlo Tree Search

GENERATED TEXT (step 3):
>> Monte Carlo Tree Search is an algorithm

EXPECTED: Monte Carlo Tree Search (MCTS) is an algorithm that combines tree search with random sampling for decision making.

AVAILABLE ACTIONS:
  0: 'that'
  1: 'combines'
  2: 'tree'
  3: 'search'
  4: 'with'
  ...and 5 more options

CURRENT REWARD: 0.234
============================================================
```

## Advanced Features

### Knowledge Graph

The knowledge graph stores semantic relationships between concepts, allowing the system to make more informed decisions based on world knowledge. Enable it with the `--use_knowledge` flag.

### Parallel MCTS

For faster processing, the system can distribute MCTS simulations across multiple processes. This is particularly useful for deeper search or when training on larger datasets. Enable with the `--parallel` flag.

### Enhanced Neural Network

The enhanced architecture includes residual connections and multi-head attention mechanisms for better performance on complex tasks. Enable with the `--enhanced` flag.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.