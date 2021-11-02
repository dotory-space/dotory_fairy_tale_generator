from transformers import PreTrainedTokenizerFast

def get_tokenizer():
    return PreTrainedTokenizerFast.from_pretrained(
        'skt/kogpt2-base-v2',
        bos_token='</s>',
        eos_token='</s>',
        unk_token='<unk>',
        pad_token='<pad>',
        mask_token='<mask>',
    )
    