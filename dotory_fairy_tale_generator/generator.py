from hanspell import spell_checker
import torch
import kss

def generate_sentences(model, tokenizer, input_sentence, device):
    encoded = torch.tensor([tokenizer.bos_token_id] + tokenizer.encode(input_sentence) + [tokenizer.eos_token_id]).unsqueeze(0)
    generated = model.generate(encoded, do_sample=True, num_return_sequences=3, max_length=60, min_length=1, temperature=0.6,
                                bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)  # length_penalty=10,  min_length=len(encoded[0])+3, # .to(device)
    output_sentence = [tokenizer.decode(generated[i], skip_special_tokens=True) for i in range(3)]  # decode
    output_sentence = [output_sentence[i].replace(input_sentence, '').lstrip(' ') for i in range(3)]  # input 중복 문장 제거
    output_sentence = [kss.split_sentences(output_sentence[i])[0] for i in range(3)]  # 첫 번째 문장 분리
    output_sentence = [spell_checker.check(output_sentence[i]).checked for i in range(3)]  # 맞춤법 검사
    output_sentence = [output_sentence[i] + '.' if output_sentence[i] and output_sentence[i][-1] in ['다','요'] else output_sentence[i] for i in range(3)]  # 마침표 추가
    return output_sentence
    