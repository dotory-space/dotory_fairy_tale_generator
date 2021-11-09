import torch
from transformers import GPT2LMHeadModel, GPT2Config

def get_model(checkpoint_path, config_file_path, device):
    config = GPT2Config.from_json_file(config_file_path)
    model = GPT2LMHeadModel(config=config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    return model

        