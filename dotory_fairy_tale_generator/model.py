import torch
from transformers import GPT2LMHeadModel, GPT2Config

def get_model(checkpoint_path, config_file_path, device):
    model = GPT2LMHeadModel(config=GPT2Config.from_json_file(config_file_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    model = GPT2LMHeadModel(GPT2Config.from_json_file(config_path))
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    return model

        