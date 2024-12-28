import asyncio
from huggingface_hub import login

from CoT_Decoding.main import main
from src.config import Config

if __name__ == "__main__":
    config = Config()
    login(token=config.hugging_face_token)
    
    asyncio.run(main(model_name=config.model_name, dataset_path=config.dataset_path, decoding_mode=config.decoding_mode, baseline_cot=config.baseline_cot))
