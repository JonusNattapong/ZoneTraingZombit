# ZoneTraingZombit - MCTS with Neural Networks

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

## Training
```bash
python MCTS_Neural_Networks/main.py --mode train --enhanced
```