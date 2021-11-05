import torch
from transformers import GPT2LMHeadModel, GPT2Config
import nltk

def get_model(checkpoint_path, config_path, device):
    checkpoint = torch.load(checkpoint_path)

    model = GPT2LMHeadModel(GPT2Config.from_json_file(config_path))
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    nltk.download('punkt')

    return model

        