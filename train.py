import json

import pandas as pd
import torch
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from model.bilstm_crf import BiLSTMCRF
from model.dataset import LogDataset
from preprocess.utils import len2mask, pad

DATA_FILE_PATH = './datasets/apache/input_ids.csv'
VOCAB_PATH = './vocab/vocab_full.json'
TAG_VOCAB_PATH = './vocab/tag_vocab.json'

epochs = 5
batch_size = 32
max_grad_norm = 5
lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collate_fn(batch, return_tensor=True):
    b_input_ids, b_tag_ids = [t[0] for t in batch], [t[1] for t in batch]

    seq_lens = [len(inputs) for inputs in b_input_ids]
    b_masks = len2mask(seq_lens)

    # pad seq in bath to same length
    max_len = max(seq_lens)
    b_input_ids = [pad(input_ids, 0, max_len) for input_ids in b_input_ids]
    b_tag_ids = [pad(input_labels, 0, max_len)
                 for input_labels in b_tag_ids]

    if return_tensor:
        return torch.LongTensor(b_input_ids), torch.LongTensor(b_tag_ids), torch.LongTensor(b_masks)
    else:
        return b_input_ids, b_tag_ids, b_masks


tag2idx: dict
with open(TAG_VOCAB_PATH, 'r') as f:
    tag2idx = json.load(f)
idx2tag = {i: tag for tag, i in tag2idx.items()}
vocab: dict
with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)

df = pd.read_csv(DATA_FILE_PATH, converters={
                 'x_ids': eval, 'y_ids': eval}, index_col=0)
input_ids, tag_ids = df['x_ids'].tolist(), df['y_ids'].tolist()

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(
    input_ids, tag_ids, random_state=1234, test_size=0.1)

train_data = LogDataset(tr_inputs[:256], tr_tags[:256])
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=batch_size, collate_fn=collate_fn)

val_data = LogDataset(val_inputs, val_tags)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(
    val_data, sampler=val_sampler, batch_size=batch_size, collate_fn=collate_fn)

# model = BertForTokenClassification.from_pretrained(
#    "bert-base-cased", num_labels=len(tag2idx))
model = BiLSTMCRF(len(vocab), len(tag2idx))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# train loop
for _ in trange(epochs, desc="Epoch"):
    count, report_every = 0, 100,
    model.train()
    tr_loss, n_sentences = 0, 0

    for step, batch in enumerate(tqdm(train_dataloader)):
        b_input_ids, b_tag_ids, b_masks = batch

        # to tensor
        b_input_ids = b_input_ids.to(device)
        b_tag_ids = b_tag_ids.to(device)
        b_masks = b_masks.to(device)

        loss = model(b_input_ids, b_tag_ids, b_masks)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        # track train loss
        tr_loss += loss.item()
        n_sentences += b_input_ids.shape[0]

        count += 1
        if count % report_every == 0:
            print(f'{count}/{len(train_dataloader)} loss: {tr_loss/n_sentences}')

    print(f'Train Loss: {tr_loss/n_sentences}')

    # VALIDATION on validation set
    model.eval()

    pred_tags, true_tags = [], []
    val_loss, val_f1 = 0, 0
    n_sentences = 0

    for batch in val_dataloader:
        b_input_ids, b_tag_ids, b_masks = batch

        # to tensor
        b_input_ids = b_input_ids.to(device)
        b_tag_ids = b_tag_ids.to(device)
        b_masks = b_masks.to(device)

        with torch.no_grad():
            loss = model(b_input_ids, b_tag_ids, b_masks)
            b_pred_ids = model.predict(b_input_ids, b_masks)

        # record loss
        val_loss += loss.item()
        n_sentences += b_input_ids.shape[0]

        # remove <pad> in true tag ids
        sen_lengths = torch.count_nonzero(b_masks, dim=1).tolist()
        b_true_ids = [ids[:length]
                      for ids, length in zip(b_tag_ids.tolist(), sen_lengths)]
        # ids to tag
        tmp_pred_tags = [[idx2tag[idx] for idx in ids] for ids in b_pred_ids]
        tmp_true_tags = [[idx2tag[idx] for idx in ids] for ids in b_true_ids]

        # replace 'X' with 'O'
        tmp_pred_tags = [
            ['O' if tag[0] == 'X' else tag for tag in tags] for tags in tmp_pred_tags]
        tmp_true_tags = [
            ['O' if tag[0] == 'X' else tag for tag in tags] for tags in tmp_true_tags]

        # exclude <START> and <END>
        pred_tags.extend(tmp_pred_tags[1:-1])
        true_tags.extend(tmp_true_tags[1:-1])

    print(f'Avg Validation loss: {val_loss/n_sentences}')
    print(f'Validation F1: {f1_score(true_tags,pred_tags,scheme=IOB2)}')
    classification_report(true_tags, pred_tags, scheme=IOB2)

pass
