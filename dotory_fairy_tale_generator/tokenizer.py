from transformers import GPT2Tokenizer

def get_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

    return tokenizer
    