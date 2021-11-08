from transformers import PreTrainedTokenizerFast

def get_tokenizer():
    print('[fairy-tale-generator] [get_tokenizer] [1]')
    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    print('[fairy-tale-generator] [get_tokenizer] [1]')
    return tokenizer
    