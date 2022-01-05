import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import nltk
from nltk.tokenize import sent_tokenize
import requests
from .hanspell import spell_checker

class FairyTaleGenerator:
    def __init__(self, checkpoint_path, tokenizer_dir_path, config_file_path, filtering_file_path):
        print('[FTG] initialize')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('[FTG] device: ', self.device)
        checkpoint = torch.load(checkpoint_path, map_location = self.device)
        print('[FTG] checkpoint loaded')
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir_path)
        print('[FTG] tokenizer: ', self.tokenizer)
        config = GPT2Config.from_json_file(config_file_path)
        print('[FTG] config: ', config)
        self.model = GPT2LMHeadModel(config)
        print('[FTG] model: ', self.model)
        self.model.load_state_dict(checkpoint['model'])
        print('[FTG] model loaded')
        self.model.to(self.device)
        nltk.download('punkt')

        f = open(filtering_file_path, 'r')
        self.filtering = f.read().split('\n')
        f.close()

    def translate_kakao(self, text, source, target):
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

    def generate(self, input_sentence):
        input_sentence = self.translate_kakao(input_sentence, 'kr', 'en')  # papago : ko, kakao : kr
        encoded = torch.LongTensor().to(self.device)

        encoded = torch.cat([encoded, torch.tensor(self.tokenizer.encode(input_sentence)).to(self.device)])
        print('Length of encoded:', len(encoded))
        generated = self.model.generate(encoded.unsqueeze(0), do_sample=True, use_cache=True, top_p=0.9, num_return_sequences=3, max_length=len(encoded)+100, min_length=len(encoded), temperature=0.6, pad_token_id=tokenizer.eos_token_id).to(device)  # length_penalty=10,
        generated = [generated[i][len(encoded):] for i in range(3)]  # input 중복 제거
        decoded = [self.tokenizer.decode(generated[i]) for i in range(3)]  # decode
        output_eng = [sent_tokenize(decoded[i])[0] if decoded[i] else decoded[i] for i in range(3)]  # 첫 번째 문장 분리, kakao 번역이 문장 단위로 잘라주기 때문에 translate_kakao에 decodede 그대로 들어감. 얘는 그저 output_eng를 위함
        output_kor = [self.translate_kakao(decoded[i], 'en', 'ku') for i in range(3)] # papgo : ko, kakao : ku (ku : 높임말 문체 in kakao)
        output_kor = [spell_checker.check(output_kor[i]).checked for i in range(3)]  # 맞춤법 검사
        for i in range(3):
            if sum([f in output_kor[i] for f in self.filtering]):  # filter it and new generate
                generated = self.model.generate(encoded.unsqueeze(0), do_sample=True, use_cache=True, top_p=0.9, num_return_sequences=1, max_length=len(encoded)+100, min_length=len(encoded), temperature=0.6, pad_token_id=tokenizer.eos_token_id).to(device)
                generated = generated[0][len(encoded):]
                decoded = self.tokenizer.decode(generated)
                try:output_eng[i] = sent_tokenize(decoded)[0]
                except:output_eng[i] = decoded
                output_kor[i] = self.translate_kakao(output_eng[i], 'en', 'ku')
                output_kor[i] = spell_checker.check(output_kor[i]).checked
            print('Generated ' + str(i+1) + ' : ', output_kor[i])

        # input_sentence = self.translate_kakao(input_sentence, 'kr', 'en')  # papago : ko, kakao : kr
        # encoded = torch.tensor(self.tokenizer.encode(input_sentence)).unsqueeze(0).to(self.device)
        # generated = self.model.generate(encoded, do_sample=True, top_p=0.9, num_return_sequences=3, max_length=200, min_length=1, temperature=0.6, pad_token_id=self.tokenizer.eos_token_id).to(self.device)  # length_penalty=10, 
        # decoded = [self.tokenizer.decode(generated[i]) for i in range(3)]  # decode
        # output_eng = [decoded[i].replace(input_sentence, '').lstrip(' ') for i in range(3)]  # input 중복 문장 제거
        # output_eng = [sent_tokenize(output_eng[i])[0] for i in range(3)]  # 첫 번째 문장 분리
        # output_kor = [self.translate_kakao(output_eng[i], 'en', 'ku') for i in range(3)]  # papgo : ko, kakao : ku (ku : 높임말 문체 in kakao)
        # output_kor = [spell_checker.check(output_kor[i]).checked for i in range(3)]  # 맞춤법 검사

        return output_kor
