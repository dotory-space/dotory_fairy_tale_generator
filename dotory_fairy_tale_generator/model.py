import torch
from transformers import GPT2LMHeadModel, GPT2Config

def get_model(checkpoint_path, config_file_path):
    print('[fairy-tale-generator] [get-model] [1]')
    config = GPT2Config.from_json_file(config_file_path)
    print('[fairy-tale-generator] [get-model] [2]')
    model = GPT2LMHeadModel(config=config)
    print('[fairy-tale-generator] [get-model] [3]')
    checkpoint = torch.load(checkpoint_path)
    print('[fairy-tale-generator] [get-model] [4]')
    model.load_state_dict(checkpoint)
    print('[fairy-tale-generator] [get-model] [5]')

    return model

        