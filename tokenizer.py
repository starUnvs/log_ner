import json
import unicodedata
from collections import defaultdict
from glob import glob

import regex as re
from tqdm import tqdm

MAX_LEN = 100000

class Tokenizer():
    def __init__(self, whitespace_token='⸏', digit_token='χ', number_token='<NUM>', chinese_token='Ĉ'):
        self.whitespace_token = whitespace_token
        self.digit_token = digit_token
        self.number_token = number_token
        self.chinese_token = chinese_token

        self.vocab_history = defaultdict(set)

    def tokenize(self, text, postprocess=True):
        text = text.strip()
        text = self._preprocess(text)

        tokens = self._split(text)

        if postprocess:
            tokens = self._postprocess(tokens)

        return tokens

    def _preprocess(self, text):
        '''
        replace whitespace with token
        insert whitespace to camel case function name, number and word combination, chinese characters
        '''
        text = text.replace(' ', self.whitespace_token)

        pt_func = re.compile(
            r'((?<=[a-z])(?=[A-Z]))|((?<=[A-Z])(?=[A-Z][a-z]))')
        pt_num2word = re.compile(
            r'(?<=[^0-9a-zA-Z]|^)(\d+)([a-zA-Z]+)(?=[^0-9a-zA-Z])|$')
        pt_word2num = re.compile(
            r'(?<=[^0-9a-zA-Z]|^)([a-zA-Z]+)(\d+)(?=[^0-9a-zA-Z])|$')
        pt_chinese = re.compile(r'([\u4e00-\u9fa5])')

        text = pt_func.sub(r' ', text)
        text = pt_num2word.sub(r'\1 \2', text)
        text = pt_word2num.sub(r'\1 \2', text)
        text = pt_chinese.sub(r' \1 ', text)

        return text

    def _split(self, text):
        '''Splits punctuation, whitespace_token and whitespace on a piece of text.'''
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char) or char == self.whitespace_token:
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        tokens = ["".join(x) for x in output]
        return ' '.join(tokens).split()

    def _postprocess(self, tokens):
        '''
        replace long number with <NUM> and short number with digit_token
        '''

        pt_num = re.compile(r'(^0x[0-9a-fA-F]+$)|(^\d{5,}$)')
        pt_random = re.compile(r'(?=(.*\d){2,})(?=.*[a-zA-Z]).*')

        pt_chinese_char = re.compile(r'([\u4e00-\u9fa5])')
        pt_all_digit = re.compile(r'^\d+$')
        pt_digit = re.compile(r'\d')

        for i, token in enumerate(tokens):
            tokens[i] = pt_num.sub(self.number_token, token)
            if tokens[i] != token:
                self.vocab_history['NUM'].add(token)
                continue

            tokens[i] = pt_random.sub('<UKN>', token)
            if tokens[i] != token:
                self.vocab_history['UKN'].add(token)
                continue

            #token = pt_chinese_char.sub(self.chinese_token, token)

            if pt_all_digit.match(token) != None:
                tokens[i] = pt_digit.sub(self.digit_token, token)
                self.vocab_history['DIGIT'].add(token)

        return tokens

    def _is_punctuation(self, char):
        """Checks whether `char` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False


def get_vocab(files):
    t = Tokenizer()

    vocab = defaultdict(int)
    for f_path in files:
        with open(f_path, errors='ignore') as f:
            print('parsing '+f_path+'...')
            for line in tqdm(f.readlines()[:MAX_LEN]):
                words = t.tokenize(line)
                for word in words:
                    vocab[word] += 1

        with open('./vocab_tmp.json', 'w') as f:
            json.dump(vocab, f, ensure_ascii=False)
    with open('./vocab_his.json', 'w') as f:
        json.dump(t.vocab_history, f, ensure_ascii=False)

    return vocab


if __name__ == '__main__':
    # files=glob('/data/logs_data/sys_data/*/*.log')
    files = glob('./datasets/*/*.log')
    vocab = get_vocab(files)
    # vocab=get_vocab(glob('./datasets/kafka/kafka_10k.log'))
    vocab = {k: v for k, v in sorted(
        vocab.items(), key=lambda x: x[1], reverse=True)}

    with open('./vocab.json', 'w') as f:
        json.dump(vocab, f, ensure_ascii=False)

    pass