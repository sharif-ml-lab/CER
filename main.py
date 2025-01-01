from huggingface_hub import login

from src.evaluation import run_dataset
from src.config import Config

if __name__ == "__main__":
    config = Config()
    login(token=config.hugging_face_token)
    
    run_dataset(config=config)
