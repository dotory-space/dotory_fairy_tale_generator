import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import nltk
from nltk.tokenize import sent_tokenize
from .hanspell import spell_checker
import pandas as pd
from pyjosa.josa import Josa
import random

class FairyTaleGenerator:
    def __init__(self, checkpoint_path, first_sentence_file_path, filtering_file_path):
        print('[FTG] initialize')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('[FTG] device: ', self.device)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [
                '<우주>', '<숲속>', '<바다>', '<마을>', '<왕국>', '<기타>', '<등장인물1>', '<등장인물2>'
            ]
        })
        print('[FTG] tokenizer: ', self.tokenizer)
        self.model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        print('[FTG] model: ', self.model)
        nltk.download('punkt')

        filtering_file = open(filtering_file_path, 'r')
        self.filtering = filtering_file.read().split('\n')
        filtering_file.close()
        print('[FTG] filtering loaded: ', self.filtering)

        first_sentence_file = open(first_sentence_file_path, 'r')
        first_sentence = first_sentence_file.read().split('\n')
        first_sentence_file.close()
        first_sentence.remove('')
        first_sentence = pd.Series(first_sentence)
        self.first_sentence_df = pd.concat([first_sentence.str[:4], first_sentence.str[5:]], axis=1, keys=['theme', 'first_sentence'])
        print('[FTG] first sentences loaded: ', )
    
    def replace_name(self, sentence, character1, character2):
        if '> ' in sentence:
            sentence = sentence.replace('> ', '>')

        if '<등장인물1>' in sentence or '<등장인물 1>' in sentence:
            sentence = sentence.replace('<등장인물1>', character1)
            sentence = sentence.replace('<등장인물 1>', character1)
            idx = sentence.index(character1)+len(character1)
            if sentence[idx] in ['을', '를', '은', '는', '이', '가', '과', '와', '이나', '나', '으로', '로', '아', '야',
                                                                    '이랑', '랑', '이며', '며', '이다', '다', '이가', '가']:
                sentence = list(sentence)
                sentence[idx] = Josa.get_josa(character1, sentence[idx])
                sentence = "".join(sentence)

        if '<등장인물2>' in sentence or '<등장인물 2>' in sentence:
            sentence = sentence.replace('<등장인물2>', character2)
            sentence = sentence.replace('<등장인물 2>', character2)
            idx = sentence.index(character2)+len(character2)
            if sentence[idx] in ['을', '를', '은', '는', '이', '가', '과', '와', '이나', '나', '으로', '로', '아', '야',
                                                                    '이랑', '랑', '이며', '며', '이다', '다', '이가', '가']:
                sentence = list(sentence)
                sentence[idx] = Josa.get_josa(character2, sentence[idx])
                sentence = "".join(sentence)
        return sentence

    def generate_first_sentence(self, theme, character1_name, character2_name):
        first_sentence_theme = self.first_sentence_df[self.first_sentence_df.theme == theme].first_sentence.values
        input_sentence = random.choice(first_sentence_theme)
        return self.replace_name(input_sentence, character1_name, character2_name)

    def generate_sentence(self, input_sentence, character1_name, character2_name):
        encoded = torch.cat([encoded, torch.tensor(self.tokenizer.encode(input_sentence)).to(self.device)])
        generated = self.model.generate(encoded.unsqueeze(0), do_sample=True, use_cache=True, top_p=0.9, \
                                num_return_sequences=3, max_length=len(encoded)+100, min_length=len(encoded), \
                                temperature=0.6, pad_token_id=self.tokenizer.eos_token_id).to(self.device)
        generated = [generated[i][len(encoded):] for i in range(3)]  # input 중복 제거
        decoded = [self.tokenizer.decode(generated[i], clean_up_tokenization_spaces=True) for i in range(3)]  # decode
        output = [sent_tokenize(decoded[i])[0] if decoded[i] else decoded[i] for i in range(3)]  # 첫번째 문장 분리
        output = [self.replace_name(output[i], character1_name, character2_name) for i in range(3)]  # 등장인물 스위칭
        output = [spell_checker.check(output[i]).checked for i in range(3)]  # 맞춤법 검사
        for i in range(3):
            if sum([f in output[i] for f in self.filtering]):  # filter it and new generate
                generated = self.model.generate(encoded.unsqueeze(0), do_sample=True, use_cache=True, top_p=0.9, \
                                    num_return_sequences=1, max_length=len(encoded)+100, min_length=len(encoded), \
                                    temperature=0.6, pad_token_id=self.tokenizer.eos_token_id).to(self.device)
                generated = generated[0][len(encoded):]
                decoded = self.tokenizer.decode(generated, clean_up_tokenization_spaces=True)
                try:output[i] = sent_tokenize(decoded)
                except:output[i] = decoded
                output[i] = self.replace_name(output[i], character1_name, character2_name)
                output[i] = spell_checker.check(output[i]).checked

        return output
