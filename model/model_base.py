from abc import abstractmethod
import json

import torch
import torch.nn as nn
from seqeval.metrics.sequence_labeling import classification_report, f1_score
from seqeval.scheme import IOB2

from model.utils import merge, align_two_seq
from model.crf import CRF


class NERModelBase(nn.Module):
    def __init__(self, embedding, encoder, char_level=True, tokenizer=None, word2idx=None, char2idx=None, tag2idx=None):
        """ Initialize the model
        Args:
            sent_vocab (Vocab): vocabulary of words
            tag_vocab (Vocab): vocabulary of tags
            embed_size (int): embedding size
            hidden_size (int): hidden state size
        """
        super().__init__()
        self.char_level = char_level
        self.tokenizer = tokenizer

        self.word2idx = word2idx
        self.idx2word = {i: key for key, i in word2idx.items()}
        self.tag2idx = tag2idx
        self.idx2tag = {i: key for key, i in tag2idx.items()}
        self.char2idx = char2idx

        num_tags = len(self.tag2idx)

        self.embedding = embedding
        self.encoder = encoder
        self.decoder = CRF(num_tags, True)

    @abstractmethod
    def forward(self, b_word_ids, b_masks, b_char_ids, b_tag_ids):
        pass

    def _convert_sentence_to_ids(self, raw_sentence):
        tokens = self.tokenizer.tokenize(raw_sentence)
        raw_tokens = self.tokenizer.tokenize(raw_sentence)
        chars = [list(w) for w in raw_tokens]

        token_ids = [self.word2idx.get(token, self.word2idx['<UNK>'])
                     for token in tokens]
        char_ids = [[self.char2idx[c] for c in w if c in self.char2idx]
                    for w in chars]
        char_maxlen = max([len(w) for w in char_ids])
        char_ids = [w+[0]*(char_maxlen-len(w)) for w in char_ids]

        mask = [1]*len(token_ids)
        return token_ids, mask, char_ids

    @abstractmethod
    def _predict(self, b_word_ids, b_masks, b_char_ids):
        pass

    def predict_raw(self, raw_sentence):
        device = next(self.parameters()).device

        word_ids, mask, char_ids = self._convert_sentence_to_ids(raw_sentence)
        b_word_ids = torch.LongTensor(word_ids).unsqueeze(0).to(device)
        b_masks = torch.LongTensor(mask).unsqueeze(0).to(device)
        b_char_ids = torch.LongTensor(char_ids).unsqueeze(0).to(device)

        tag_ids = self._predict(b_word_ids, b_masks, b_char_ids)[0]
        tags = [self.idx2tag[idx] for idx in tag_ids]

        origin_words = self.tokenizer.tokenize(raw_sentence, postprocess=False)
        merged_words, merged_tags = merge(origin_words, tags)

        return merged_words, merged_tags

# TODO:
    def test(self, dataloader, verbose=True, output_file=None):
        self.eval()
        device = next(self.parameters()).device

        val_loss, n_sentences = 0, 0

        sent_ids, pred_ids, true_ids = [], [], []
        # one epoch
        for batch in dataloader:

            b_word_ids, b_masks, b_char_ids, b_tag_ids = batch

            # to device
            b_word_ids = b_word_ids.to(device)
            b_masks = b_masks.to(device)
            b_char_ids = b_char_ids.to(device)
            b_tag_ids = b_tag_ids.to(device)

            with torch.no_grad():
                loss = self(b_word_ids, b_masks, b_char_ids, b_tag_ids)
                b_pred_ids = self._predict(b_word_ids, b_masks, b_char_ids)

            # record loss
            val_loss += loss.item()
            n_sentences += b_word_ids.shape[0]

            # remove <pad> in true tag ids
            sen_lengths = torch.count_nonzero(b_masks, dim=1).tolist()
            b_sent_ids = [ids[:length]
                          for ids, length in zip(b_word_ids.tolist(), sen_lengths)]
            b_true_ids = [ids[:length]
                          for ids, length in zip(b_tag_ids.tolist(), sen_lengths)]

            sent_ids.extend(b_sent_ids)
            pred_ids.extend(b_pred_ids)
            true_ids.extend(b_true_ids)

        # ids to tag
        all_pred_tags = [[self.idx2tag[idx]
                          for idx in ids] for ids in pred_ids]
        all_true_tags = [[self.idx2tag[idx]
                          for idx in ids] for ids in true_ids]

        # replace 'X' with 'O'
        all_pred_tags = [
            ['O' if tag[0] == 'X' else tag for tag in tags] for tags in all_pred_tags]
        all_true_tags = [
            ['O' if tag[0] == 'X' else tag for tag in tags] for tags in all_true_tags]

        avg_loss = val_loss/n_sentences
        f1 = f1_score(all_true_tags, all_pred_tags)
        f1_strict = f1_score(all_true_tags, all_pred_tags,
                             mode='strict', scheme=IOB2)

        if verbose:
            print(f'Avg loss: {avg_loss}, F1: {f1}, F1 STRICT: {f1_strict}')
            print(classification_report(
                all_true_tags, all_pred_tags, zero_division=0))
            print(classification_report(
                all_true_tags, all_pred_tags, zero_division=0, scheme=IOB2, mode='strict'))

        return avg_loss, f1, f1_strict
