import unicodedata
from collections import defaultdict
import json

import regex as re


class BaseTokenizer():
    def __init__(self, unk_token='<UNK>', digit_token='<d>', number_token='<NUM>', chinese_token='<c>'):
        self.unk_token = unk_token
        self.digit_token = digit_token
        self.number_token = number_token
        self.chinese_token = chinese_token

        self.vocab_history = defaultdict(set)

    def _tokenize(self, sentence, postprocess=True):
        sentence = sentence.strip('\n')
        sentence = self._preprocess(sentence)

        tokens = self._split(sentence)

        if postprocess:
            tokens = self._postprocess(tokens)

        return tokens

    def tokenize(self, text, postprocess=True):
        if type(text) is list:
            return [self._tokenize(sent,postprocess) for sent in text]
        elif type(text) is str:
            return self._tokenize(text,postprocess)
        else:
            raise ValueError('type error')

    def _preprocess(self, text):
        '''
        replace whitespace with token
        insert whitespace to camel case function name, number and word combination, chinese characters
        '''
        #text = text.replace(' ', self.whitespace_token)

        pt_func = re.compile(
            r'((?<=[a-z])(?=[A-Z]))|((?<=[A-Z])(?=[A-Z][a-z]))')
        pt_num2word = re.compile(
            r'(?<=[^0-9a-zA-Z]|^)(\d+)([a-zA-Z]+)(?=[^0-9a-zA-Z]|$)')
        pt_word2num = re.compile(
            r'(?<=[^0-9a-zA-Z]|^)([a-zA-Z]+)(\d+)(?=[^0-9a-zA-Z]|$)')
        pt_chinese = re.compile(r'([\u4e00-\u9fa5])')

        text = pt_func.sub(r'^', text)
        text = pt_num2word.sub(r'\1^\2', text)
        text = pt_word2num.sub(r'\1^\2', text)
        text = pt_chinese.sub(r'\1^', text)

        return text

    def _split(self, text):
        '''Splits punctuation, whitespace_token and whitespace on a piece of text.'''
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char) or char == ' ':
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        tokens = ["".join(x) for x in output]
        return [token for token in tokens if token != '^']

    def _postprocess(self, tokens):
        '''
        replace long number with <NUM> and short number with digit_token
        '''

        pt_num = re.compile(r'(^0x[0-9a-fA-F]+$)|(^\d{5,}$)')
        pt_random = re.compile(r'(?=(.*\d){2,})(?=.*[a-zA-Z]).*')

        # pt_chinese_char = re.compile(r'([\u4e00-\u9fa5])')
        pt_all_digit = re.compile(r'^\d+$')
        pt_digit = re.compile(r'\d')

        for i, token in enumerate(tokens):
            tokens[i] = pt_num.sub(self.number_token, token)
            if tokens[i] != token:
                self.vocab_history['NUM'].add(token)
                continue

            tokens[i] = pt_random.sub(self.unk_token, token)
            if tokens[i] != token:
                self.vocab_history['UNK'].add(token)
                continue

            # token = pt_chinese_char.sub(self.chinese_token, token)

            if pt_all_digit.match(token) is not None:
                tokens[i] = pt_digit.sub(self.digit_token, token)
                self.vocab_history['NUM'].add(token)

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


if __name__ == '__main__':
    t = BaseTokenizer()
    tokens = t.tokenize(
        '2015-10-18 18:07:30,627 ERROR [RMCommunicator Allocator] org.apache.hadoop.mapreduce.v2.app.rm.RMContainerAllocator: ERROR IN CONTACTING RM.')

    pass
