import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json

from model.crf import CRF
from model.model_base import NERModelBase
from model.tokenizer import BaseTokenizer


class WordWithCharEmbedding(nn.Module):
    def __init__(self, num_words, num_chars, word_embedding_size, char_embedding_size, char_conv_size):
        super().__init__()
        self.word_embedding = nn.Embedding(
            num_words, word_embedding_size, padding_idx=0)
        self.char_embedding = nn.Embedding(
            num_chars, char_embedding_size, padding_idx=0)
        self.conv = nn.Conv1d(char_embedding_size,
                              char_conv_size, 5, padding='same')

    def forward(self, b_word_ids, b_char_ids):
        #        b_word_ids = []
        #        b_char_ids = []
        #        for words in b_input_words:
        #            b_word_ids.append([self.word2idx[word] for word in words])
        #            words = [list(word) for word in words]
        #            char_ids = [self.char2idx[char] for char in word for word in words]
        #            b_char_ids.append(char_ids)
        #
        #        device = next(self.parameters()).device
        #        b_word_ids = torch.LongTensor(b_word_ids)  # shape: (b, len)
        #        b_char_ids = torch.LongTensor(b_char_ids)  # shape: (b, len, word_len)

        batch_size, seq_len = b_word_ids.shape
        # shape: (b, len, word_embedding_size)
        word_embedding = self.word_embedding(b_word_ids)

        b_char_ids = b_char_ids.view(batch_size*seq_len, -1)
        # shape (b*len, word_len, char_embedding_size)
        char_embedding = self.char_embedding(b_char_ids).permute(0, 2, 1)
        char_embedding = self.conv(char_embedding).permute(0, 2, 1)
        # shape (b*len, char_conv_size)
        char_embedding, _ = torch.max(char_embedding, axis=1)
        char_embedding = char_embedding.view(batch_size, seq_len, -1)

        return torch.cat([word_embedding, char_embedding], dim=2)


class BiLSTMEncoder(nn.Module):
    def __init__(self,  embed_size, hidden_size, num_tags):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=embed_size,
                              hidden_size=hidden_size, bidirectional=True)
        self.hidden2emit_score = nn.Linear(hidden_size * 2, num_tags)
        self.dropout = nn.Dropout(0.5)

    def forward(self, b_feats, b_masks):
        b_lengths = torch.count_nonzero(b_masks, dim=1).tolist()
        total_length = b_masks.shape[1]
        # b_inputs = self.embedding(b_inputs)  # shape: (b, len, e)

        padded_sentences = pack_padded_sequence(
            b_feats, b_lengths, batch_first=True, enforce_sorted=False)
        hidden_states, _ = self.bilstm(padded_sentences)
        hidden_states, _ = pad_packed_sequence(
            hidden_states, batch_first=True, total_length=total_length)  # shape: (b, len, 2h)
        emit_score = self.hidden2emit_score(
            hidden_states)  # shape: (b, len, K)
        emit_score = self.dropout(emit_score)  # shape: (b, len, K)
        return emit_score


class CNNEncoder(nn.Module):
    def __init__(self, in_feat_size, conv_size, fc_size, num_tags):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_feat_size, conv_size,
                                kernel_size=3, padding='same')
        self.conv_2 = nn.Conv1d(conv_size, conv_size,
                                kernel_size=3, padding='same')
        self.conv_3 = nn.Conv1d(conv_size, conv_size,
                                kernel_size=3, padding='same')
        self.conv_4 = nn.Conv1d(conv_size, conv_size,
                                kernel_size=3, padding='same')
        self.conv_5 = nn.Conv1d(conv_size, conv_size,
                                kernel_size=3, padding='same')
        self.conv = nn.ModuleList(
            [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5])

        self.fc1 = nn.Linear(conv_size, fc_size)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(fc_size, num_tags)

    def forward(self, embed_feat):
        feats = embed_feat.permute(0, 2, 1)
        for conv in self.conv:
            feats = conv(feats)
        feats = feats.permute(0, 2, 1)
        feats = self.dropout1(self.fc1(feats))
        feats = self.dropout2(self.fc2(feats))

        return self.fc3(feats)


class BiLSTMLogNER(NERModelBase):
    def __init__(self,
                 char_level=True,
                 word_embed_size=128,
                 char_embed_size=10,
                 char_feat_size=50,
                 hidden_size=128,
                 tokenizer=None,
                 word2idx=None,
                 char2idx=None,
                 tag2idx=None):
        self.char_level = char_level
        if char_level:
            embedding = WordWithCharEmbedding(
                len(word2idx), len(char2idx), word_embed_size, char_embed_size, char_feat_size)
            embed_size = word_embed_size+char_feat_size
        else:
            embedding = nn.Embedding(len(word2idx), word_embed_size)
            embed_size = word_embed_size

        encoder = BiLSTMEncoder(embed_size=embed_size,
                                hidden_size=hidden_size, num_tags=len(tag2idx))

        super().__init__(embedding, encoder, char_level,
                         tokenizer, word2idx, char2idx, tag2idx)

    def forward(self, b_word_ids, b_masks, b_char_ids, b_tag_ids):
        """
        Args:
            b_input_ids (tensor): sentences, shape (b, len).
            b_tag_ids (tensor): corresponding tags, shape (b, len)
            b_masks (list): shape (b, len).
        Returns:
            loss (tensor): loss on the batch, shape (1,)
        """
        if self.char_level:
            embed_feat = self.embedding(b_word_ids, b_char_ids)
        else:
            embed_feat = self.embedding(b_word_ids)
        emit_score = self.encoder(embed_feat, b_masks)

        llk = self.decoder(emit_score, b_tag_ids,
                           mask=b_masks.byte())  # shape: (1,)
        return -llk

    def _predict(self, b_word_ids, b_masks, b_char_ids):
        """
        Args:
            b_input_ids (tensor): sentences, shape (b, len). Lengths are in decreasing order, len is the length
                                of the longest sentence
            b_masks (list[list]): masks
        Returns:
            tags (list[list[int]]): predicted index of tags for the batch
        """
        if self.char_level:
            embed_feats = self.embedding(b_word_ids, b_char_ids)
        else:
            embed_feats = self.embedding(b_word_ids)
        emit_score = self.encoder(
            embed_feats, b_masks)  # shape: (b, len, K)
        tag_ids = self.decoder.decode(emit_score, mask=b_masks.byte())

        return tag_ids


class CNNLogNER(NERModelBase):
    def __init__(self, char_level=True,
                 word_embed_size=128,
                 char_embed_size=10,
                 char_feat_size=50,
                 conv_size=256,
                 fc_size=1024,
                 tokenizer=None,
                 word2idx=None,
                 char2idx=None,
                 tag2idx=None):
        self.char_level = char_level
        if char_level:
            embedding = WordWithCharEmbedding(
                len(word2idx), len(char2idx), word_embed_size, char_embed_size, char_feat_size)
            embed_size = word_embed_size+char_feat_size
        else:
            embedding = nn.Embedding(len(word2idx), word_embed_size)
            embed_size = word_embed_size
        encoder = CNNEncoder(embed_size, conv_size, fc_size, len(tag2idx))

        super().__init__(
            embedding, encoder, char_level, tokenizer, word2idx, char2idx, tag2idx)

    def forward(self, b_word_ids, b_masks, b_char_ids, b_tag_ids):
        """
        Args:
            b_input_ids (tensor): sentences, shape (b, len).
            b_tag_ids (tensor): corresponding tags, shape (b, len)
            b_masks (list): shape (b, len).
        Returns:
            loss (tensor): loss on the batch, shape (1,)
        """
        if self.char_level:
            embed_feat = self.embedding(b_word_ids, b_char_ids)
        else:
            embed_feat = self.embedding(b_word_ids)
        emit_score = self.encoder(embed_feat)

        llk = self.decoder(emit_score, b_tag_ids,
                           mask=b_masks.byte())  # shape: (1,)
        return -llk

    def _predict(self, b_word_ids, b_masks, b_char_ids):
        """
        Args:
            b_input_ids (tensor): sentences, shape (b, len). Lengths are in decreasing order, len is the length
                                of the longest sentence
            b_masks (list[list]): masks
        Returns:
            tags (list[list[int]]): predicted index of tags for the batch
        """
        if self.char_level:
            embed_feats = self.embedding(b_word_ids, b_char_ids)
        else:
            embed_feats = self.embedding(b_word_ids)
        emit_score = self.encoder(
            embed_feats)  # shape: (b, len, K)
        tag_ids = self.decoder.decode(emit_score, mask=b_masks.byte())

        return tag_ids


if __name__ == '__main__':
    embedding = WordWithCharEmbedding(100, 10, 128, 10, 50)
    crf = CRF(21, True)
    word_ids = torch.LongTensor([[1, 2, 3]])
    char_ids = torch.LongTensor([[[1, 0], [1, 0], [1, 2]]])
    tag_ids = torch.LongTensor([[1, 2, 0]])
    b_masks = torch.LongTensor([[1, 1, 0]])

    embed_feat = embedding(word_ids, char_ids)
    encoder = CNNEncoder(178, 256, 1024, 21)
    emit_score = encoder(embed_feat)
    loss = crf(emit_score, tag_ids, b_masks)

    encoder = BiLSTMEncoder(178, 256, 21)
    emit_score = encoder(embed_feat, b_masks)
    loss = crf(emit_score, tag_ids, b_masks)

    tokenizer = BaseTokenizer()
    with open('/home/dell/sid/code/vocab/vocab.json', 'r') as f:
        word2idx = json.load(f)
    with open('/home/dell/sid/code/vocab/tag_vocab.json', 'r') as f:
        tag2idx = json.load(f)
    with open('/home/dell/sid/code/vocab/char_vocab.json', 'r') as f:
        char2idx = json.load(f)

    model = BiLSTMLogNER(tokenizer=tokenizer,
                         word2idx=word2idx, tag2idx=tag2idx, char2idx=char2idx)
    loss = model(word_ids, b_masks, char_ids, tag_ids)

    model = CNNLogNER(tokenizer=tokenizer, word2idx=word2idx,
                      tag2idx=tag2idx, char2idx=char2idx)
    loss = model(word_ids, b_masks, char_ids, tag_ids)

    tags = model.predict_raw('hello world')

    pass
