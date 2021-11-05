import requests
from hanspell import spell_checker
from nltk.tokenize import sent_tokenize
import torch

def translate_kakao(text, source, target):
    url = "https://translate.kakao.com/translator/translate.json"

    headers = {
        "Referer": "https://translate.kakao.com/",
        "User-Agent": "Mozilla/5.0"
    }

    data = {
        "queryLanguage": source,
        "resultLanguage": target,
        "q": text
    }

    resp = requests.post(url, headers=headers, data=data)
    data = resp.json()
    output = data['result']['output'][0][0]
    return output

def generate_sentences(model, tokenizer, input_sentence, device):
    encoded = torch.tensor(tokenizer.encode(input_sentence)).unsqueeze(0).to(device)
    generated = model.generate(encoded, do_sample=True, top_p=0.9, num_return_sequences=3, max_length=200, min_length=1, temperature=0.6, pad_token_id=tokenizer.eos_token_id).to(device)  # length_penalty=10, 
    decoded = [tokenizer.decode(generated[i]) for i in range(3)]  # decode
    output_eng = [decoded[i].replace(input_sentence, '').lstrip(' ') for i in range(3)]  # input 중복 문장 제거
    output_eng = [sent_tokenize(output_eng[i])[0] for i in range(3)]  # 첫 번째 문장 분리
    output_kor = [translate_kakao(output_eng[i], 'en', 'ku') for i in range(3)]  # papgo : ko, kakao : ku (ku : 높임말 문체 in kakao)
    output_kor = [spell_checker.check(output_kor[i]).checked for i in range(3)]  # 맞춤법 검사
    
    return output_kor
    