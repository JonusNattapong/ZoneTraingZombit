import json
import os
import requests
import concurrent.futures
import time
import argparse
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential

class DatasetGenerator:
    def __init__(self, api_key: str = None):
        """Initialize with config from environment variables"""
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key not provided and MISTRAL_API_KEY environment variable not set")
            
        self.samples = []
        self.base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        self.requests_per_minute = int(os.getenv("REQUESTS_PER_MINUTE", "30"))
        self.model_name = os.getenv("MISTRAL_MODEL", "mistral-tiny")
        self.timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
        self.last_request_time = 0
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _rate_limited_request(self):
        """Handle rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < 60/self.requests_per_minute:
            time.sleep(60/self.requests_per_minute - elapsed)
        self.last_request_time = time.time()

    def generate_sample(self, prompt: str) -> Dict:
        """Generate a sample using Mistral API"""
        self._rate_limited_request()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(os.getenv("TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("MAX_TOKENS", "100")),
            "top_p": float(os.getenv("TOP_P", "1.0"))
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            completion = response.json()["choices"][0]["message"]["content"]
            
            return {
                "prompt": prompt,
                "completion": completion,
                "metadata": {
                    "model": "mistral-tiny",
                    "temperature": 0.7
                }
            }
        except Exception as e:
            print(f"Error generating sample: {str(e)}")
            raise
    def generate_dataset(self, prompts: List[str], output_dir="datasets", min_length=50):
        """Generate high-quality dataset with parallel processing"""
        os.makedirs(output_dir, exist_ok=True)
        total_samples = len(prompts)
        quality_checks = {
            'min_length': min_length,
            'duplicates': 0,
            'short_responses': 0
        }
        total_samples = len(prompts)
        completed = 0
        failed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_prompt = {
                executor.submit(self.generate_sample, prompt): prompt
                for prompt in prompts
            }
            
            for future in concurrent.futures.as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    sample = future.result()
                    
                    # Quality checks
                    if len(sample['completion']) < quality_checks['min_length']:
                        quality_checks['short_responses'] += 1
                        continue
                        
                    if any(s['prompt'] == sample['prompt'] for s in self.samples):
                        quality_checks['duplicates'] += 1
                        continue
                        
                    self.samples.append(sample)
                    completed += 1
                    
                    # Save progress every 10 samples
                    if completed % 10 == 0:
                        self._save_dataset(output_dir, completed)
                        print(f"Progress: {completed}/{total_samples} samples "
                              f"({completed/total_samples:.1%}) | "
                              f"Failed: {failed}")
                        
                except Exception as e:
                    print(f"Error processing prompt '{prompt[:30]}...': {str(e)}")
                    failed += 1
                    continue
                    
        self._save_dataset(output_dir, completed)
        print(f"\n=== Generation Results ===")
        print(f"Completed: {completed}/{total_samples} samples")
        print(f"Failed API calls: {failed}")
        print(f"Rejected - Too short: {quality_checks['short_responses']}")
        print(f"Rejected - Duplicates: {quality_checks['duplicates']}")
        print(f"\nFinal dataset size: {len(self.samples)} samples")
        print(f"Saved to: {output_dir}/")
        print("=========================")
        
    def _save_dataset(self, output_dir, num_samples):
        """Save dataset to JSON file"""
        output_path = os.path.join(output_dir, f"dataset_{num_samples}.json")
        with open(output_path, 'w') as f:
            json.dump(self.samples, f, indent=2)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate dataset using Mistral API')
    parser.add_argument('--size', type=int, default=20,
                      help='Number of samples to generate (default: 20)')
    parser.add_argument('--min-length', type=int, default=50,
                      help='Minimum response length in characters (default: 50)')
    parser.add_argument('--output-dir', type=str, default="datasets",
                      help='Output directory for dataset (default: datasets)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Enhanced prompts for MCTS and typhoon-7b training
    Useprompts = [
        # MCTS and RL concepts
        "Explain Monte Carlo Tree Search in the context of reinforcement learning",
        "Describe how the selection phase works in MCTS",
        "What are the advantages of using neural networks with MCTS?",
        "How does the backpropagation step work in MCTS?",
        "Explain the exploration-exploitation tradeoff in MCTS",
        
        # Neural network architectures
        "Describe the architecture of a neural network suitable for MCTS value estimation",
        "What are the key components of a policy network for MCTS?",
        "How would you design a neural network to evaluate game states?",
        
        # Thai language and context
        "อธิบายการทำงานของ MCTS แบบละเอียดเป็นภาษาไทย",
        "เขียนโค้ด Python สำหรับการจำลอง MCTS พร้อมคำอธิบายเป็นภาษาไทย",
        "อธิบายแนวคิด Reinforcement Learning เป็นภาษาไทยพร้อมตัวอย่าง",
        
        # Game strategy and decision making
        "How would MCTS evaluate a chess endgame position?",
        "Describe how to adapt MCTS for real-time strategy games",
        "What modifications would MCTS need for imperfect information games?",
        
        # Technical implementation
        "Write a Python class for an MCTS node with neural network integration",
        "How would you parallelize MCTS simulations?",
        "Explain the UCB1 formula used in MCTS selection"
    ]
    
    try:
        # Initialize with environment variables
        generator = DatasetGenerator()
        print("\nConfiguration:")
        print(f"Model: {generator.model_name}")
        print(f"Base URL: {generator.base_url}")
        print(f"Max Workers: {generator.max_workers}")
        print(f"Requests/Min: {generator.requests_per_minute}")
        print(f"Timeout: {generator.timeout}s")
        print(f"Temperature: {float(os.getenv('TEMPERATURE', '0.7'))}")
        print(f"Max Tokens: {int(os.getenv('MAX_TOKENS', '100'))}")
        print(f"Top P: {float(os.getenv('TOP_P', '1.0'))}\n")
        
        print(f"Generating dataset with {len(Useprompts)} prompts...")
        generator.generate_dataset(prompts=Useprompts)
    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("\nRequired environment variables:")
        print("MISTRAL_API_KEY - Your Mistral API key")
        print("\nOptional environment variables:")
        print("MISTRAL_BASE_URL - API base URL (default: https://api.mistral.ai/v1)")
        print("MAX_WORKERS - Parallel workers (default: 4)")
        print("REQUESTS_PER_MINUTE - Rate limit (default: 30)")
        print("MISTRAL_MODEL - Model name (default: mistral-tiny)")
        print("REQUEST_TIMEOUT - Timeout in seconds (default: 30)")
        print("TEMPERATURE - Generation temperature (default: 0.7)")
        print("MAX_TOKENS - Max tokens per response (default: 100)")
        print("TOP_P - Nucleus sampling (default: 1.0)")