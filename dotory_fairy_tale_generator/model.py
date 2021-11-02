from transformers import GPT2LMHeadModel, GPT2Config

class FairyTaleGenerator(GPT2LMHeadModel):
    def __init__(self, GPT2_config_json_file_path):
        super(FairyTaleGenerator, self).__init__(config=GPT2Config.from_json_file(GPT2_config_json_file_path))
        