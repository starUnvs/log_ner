import json
from model.tokenizer import BaseTokenizer
import torch
from .utils import len2mask, load_data, pad


def prepare_pretokenized_data(words, tags, tokenizer, word2idx, char2idx, tag2idx, max_sentence_len=200, max_word_len=30):
    tokens = []
    chars = []
    token_tags = []

    for word, tag in zip(words, tags):
        tmp_tokens = tokenizer.tokenize(word)
        tokens.extend(tmp_tokens)

        origin_tokens = tokenizer.tokenize(word, postprocess=False)
        chars.extend([list(t) for t in origin_tokens])

        token_tags.append(tag)
        if len(tmp_tokens) == 1:
            continue

        if tag[0] == 'B' or tag[0] == 'I':
            post_tag = 'I'+tag[1:]
        elif tag == 'O':
            post_tag = 'X'

        token_tags.extend([post_tag]*(len(tmp_tokens)-1))

    token_ids = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    char_ids = [[char2idx[c] for c in w if c in char2idx]
                for w in chars]
    tag_ids = [tag2idx[tag] for tag in token_tags]

    mask = [1]*len(token_ids)+[0]*(max_sentence_len-len(token_ids))

    token_ids = pad(token_ids, 0, max_sentence_len)
    tag_ids = pad(tag_ids, 0, max_sentence_len)
    char_ids.extend([[]]*(max_sentence_len-len(char_ids)))
    for i, seq in enumerate(char_ids):
        char_ids[i] = pad(seq, 0, max_word_len)

    return token_ids, mask, char_ids, token_ids


class LogNERDataset(torch.utils.data.Dataset):
    def __init__(self, nwords, ntags, tokenizer, word2idx, char2idx, tag2idx):
        #        self.ntoken_ids, self.nmask, self.nchar_ids, self.ntag_ids = [], [], [], []
        #        for words, tags in zip(nwords, ntags):
        #            token_ids, mask, char_ids, tag_ids = prepare_pretokenized_data(
        #                words, tags, tokenizer, word2idx, char2idx, tag2idx)
        #            self.ntoken_ids.append(token_ids)
        #            self.nmask.append(mask)
        #            self.nchar_ids.append(char_ids)
        #            self.ntag_ids.append(tag_ids)
        self.nwords = nwords
        self.ntags = ntags
        self.tokenizer = tokenizer
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.ntoken_ids)

    def __getitem__(self, idx):
        # return self.ntoken_ids[idx], self.nmask[idx], self.nchar_ids[idx], self.ntag_ids[idx]
        return prepare_pretokenized_data(self.nwords[idx], self.ntags[idx], self.tokenizer, self.word2idx, self.char2idx, self.tag2idx)


if __name__ == '__main__':
    ntokens, ntags = load_data('/home/dell/sid/code/datasets/test_full.csv')
    tokenizer = BaseTokenizer()
    with open('/home/dell/sid/code/vocab/vocab.json', 'r') as f:
        word2idx = json.load(f)
    with open('/home/dell/sid/code/vocab/tag_vocab.json', 'r') as f:
        tag2idx = json.load(f)
    with open('/home/dell/sid/code/vocab/char_vocab.json', 'r') as f:
        char2idx = json.load(f)

    ds = LogNERDataset(ntokens, ntags, tokenizer, word2idx, char2idx, tag2idx)
    data = ds[1]

    pass
