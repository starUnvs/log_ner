from typing import List, Optional

import torch
import torch.nn as nn
from seqeval.metrics.sequence_labeling import classification_report, f1_score
from seqeval.scheme import IOB2
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from preprocess.utils import align_two_seq


class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.type_as(emissions).sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(
                f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class BiLSTMCRF(nn.Module):
    def __init__(self, embed_size=256, hidden_size=256, dropout_rate=0.5, tokenizer=None, vocab2idx=None, tag2idx=None):
        """ Initialize the model
        Args:
            sent_vocab (Vocab): vocabulary of words
            tag_vocab (Vocab): vocabulary of tags
            embed_size (int): embedding size
            hidden_size (int): hidden state size
        """
        super(BiLSTMCRF, self).__init__()
        self.tokenizer = tokenizer

        self.vocab2idx = vocab2idx
        self.idx2vocab = {i: key for key, i in vocab2idx.items()}
        self.tag2idx = tag2idx
        self.idx2tag = {i: key for key, i in tag2idx.items()}

        num_vocab = len(self.vocab2idx)
        num_tags = len(self.tag2idx)

        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_vocab, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=hidden_size, bidirectional=True)
        self.hidden2emit_score = nn.Linear(
            hidden_size * 2, num_tags)
        self.crf = CRF(num_tags, True)

    def _lstm_encode(self, b_inputs, b_masks):
        """forward through embedding layer and bilstm, returns emit_score

        Args:
            b_input_ids (tensor): shape (b, len)
            b_masks (list[list[int]]): masks, shape (b, len)

        Returns:
            [tensor]: shape (b, len, K), K is number of tags
        """
        b_lengths = torch.count_nonzero(b_masks, dim=1).tolist()
        b_inputs = self.embedding(b_inputs)  # shape: (b, len, e)

        padded_sentences = pack_padded_sequence(
            b_inputs, b_lengths, batch_first=True, enforce_sorted=False)
        hidden_states, _ = self.encoder(padded_sentences)
        hidden_states, _ = pad_packed_sequence(
            hidden_states, batch_first=True)  # shape: (b, len, 2h)
        emit_score = self.hidden2emit_score(
            hidden_states)  # shape: (b, len, K)
        emit_score = self.dropout(emit_score)  # shape: (b, len, K)
        return emit_score

    def forward(self, b_input_ids, b_tag_ids, b_masks):
        """
        Args:
            b_input_ids (tensor): sentences, shape (b, len).
            b_tag_ids (tensor): corresponding tags, shape (b, len)
            b_masks (list): shape (b, len).
        Returns:
            loss (tensor): loss on the batch, shape (1,)
        """
        emit_score = self._lstm_encode(b_input_ids, b_masks)

        llk = self.crf(emit_score, b_tag_ids,
                       mask=b_masks.byte())  # shape: (1,)
        return -llk

    def _predict(self, b_input_ids, b_masks):
        """
        Args:
            b_input_ids (tensor): sentences, shape (b, len). Lengths are in decreasing order, len is the length
                                of the longest sentence
            b_masks (list[list]): masks
        Returns:
            tags (list[list[int]]): predicted index of tags for the batch
        """
        emit_score = self._lstm_encode(
            b_input_ids, b_masks)  # shape: (b, len, K)
        tags = self.crf.decode(emit_score, mask=b_masks.byte())

        return tags

    def test(self, dataloader, verbose=True, output_file=None):

        self.eval()
        device = next(self.parameters()).device

        val_loss, n_sentences = 0, 0

        sent_ids, pred_ids, true_ids = [], [], []
        # one epoch
        for batch in tqdm(dataloader):
            b_input_ids, b_tag_ids, b_masks = batch

            # to tensor
            b_input_ids = b_input_ids.to(device)
            b_tag_ids = b_tag_ids.to(device)
            b_masks = b_masks.to(device)

            with torch.no_grad():
                loss = self(b_input_ids, b_tag_ids, b_masks)
                b_pred_ids = self._predict(b_input_ids, b_masks)

            # record loss
            val_loss += loss.item()
            n_sentences += b_input_ids.shape[0]

            # remove <pad> in true tag ids
            sen_lengths = torch.count_nonzero(b_masks, dim=1).tolist()
            b_sent_ids = [ids[:length]
                          for ids, length in zip(b_input_ids.tolist(), sen_lengths)]
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

        if output_file:
            f = open(output_file, 'w')

            sentences = [[self.idx2vocab[idx] for idx in ids]
                         for ids in sent_ids]

            for words, pred_tags, true_tags in zip(sentences, all_pred_tags, all_true_tags):
                pred_merged_words, pred_merged_tags = align_two_seq(*self._merge(
                    words, pred_tags))

                true_merged_words, true_merged_tags = align_two_seq(*self._merge(
                    words, true_tags))

                f.write('||'.join(''.join(pred_merged_words).split())+'\n')
                f.write('||'.join(''.join(pred_merged_tags).split())+'\n')
                f.write('||'.join(''.join(true_merged_words).split())+'\n')
                f.write('||'.join(''.join(true_merged_tags).split())+'\n')
                f.write('\n')
            f.close()

        # replace 'X' with 'O'
        all_pred_tags = [
            ['O' if tag[0] == 'X' else tag for tag in tags] for tags in all_pred_tags]
        all_true_tags = [
            ['O' if tag[0] == 'X' else tag for tag in tags] for tags in all_true_tags]

        # remove <START> and <END>
        all_pred_tags = [tags[1:-1] for tags in all_pred_tags]
        all_true_tags = [tags[1:-1] for tags in all_true_tags]

        avg_loss = val_loss/n_sentences
        f1 = f1_score(all_true_tags, all_pred_tags)

        if verbose:

            print(f'Avg loss: {avg_loss}')
            print(f'F1: {f1}')
            print(classification_report(all_true_tags, all_pred_tags))
            print(
                f"F1 STRICT: {f1_score(all_true_tags,all_pred_tags,mode='strict',scheme=IOB2)}")
            print(classification_report(
                all_true_tags, all_pred_tags, scheme=IOB2, mode='strict'))

        return val_loss/n_sentences, f1

    def predict_raw(self, raw_sentences):
        device = next(self.parameters()).device
        pad_idx = self.vocab2idx['<UNK>']

        words = ['<START>']+self.tokenizer.tokenize(raw_sentences)+['<END>']
        input_ids = [self.vocab2idx.get(token, pad_idx) for token in words]
        mask = [1]*len(input_ids)

        b_input_ids = torch.LongTensor([input_ids]).to(device)
        b_masks = torch.LongTensor([mask]).to(device)

        tag_ids = self._predict(b_input_ids, b_masks)[0]
        tags = [self.idx2tag[idx] for idx in tag_ids]

        merged_words, merged_tags = self._merge(words, tags)

        return merged_words, merged_tags

    def _merge(self, words, tags):
        slices = []
        start, end = 0, 1
        for i, tag in enumerate(tags[1:], 1):
            if tag[0] != 'I' and tag[0] != 'X':
                end = i
                slices.append((start, end))
                start = i
        slices.append((start, end+1))

        merged_words = []
        merged_tags = []
        for start, end in slices:
            merged_words.append(''.join(words[start:end]))
            first_tag = tags[start]
            if first_tag[0] == 'B' or first_tag[0] == 'I':
                merged_tags.append(first_tag[2:])
            else:
                merged_tags.append(first_tag)

        return merged_words, merged_tags


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BiLSTMCRF(1000, 10).to(device)

    sentences = torch.randint(0, 1000, (2, 50)).to(device)
    tags = torch.randint(0, 10, (2, 50)).to(device)
    mask = torch.IntTensor([[1]*48+[0, 0], [1]*50]).to(device)

    model(sentences, tags, mask)

    model._predict(sentences, mask)


if __name__ == '__main__':
    main()
