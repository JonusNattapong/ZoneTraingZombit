from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def download_model():
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints/typhoon-7b", exist_ok=True)
    
    print("Downloading typhoon-7b model from Hugging Face...")
    
    # Download model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("scb10x/typhoon-7b")
    tokenizer = AutoTokenizer.from_pretrained("scb10x/typhoon-7b")
    
    # Save locally
    model.save_pretrained("checkpoints/typhoon-7b")
    tokenizer.save_pretrained("checkpoints/typhoon-7b")
    
    print("Model successfully saved to checkpoints/typhoon-7b")

if __name__ == "__main__":
    download_model()