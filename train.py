import json

import pandas as pd
import torch
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from model.bilstm_crf import BiLSTMCRF
from model.dataset import LogDataset, collate_fn

DATA_FILE_PATH = './inputs/train.csv'
VOCAB_PATH = './vocab/vocab_full.json'
TAG_VOCAB_PATH = './vocab/tag_vocab.json'
MODEL_SAVE_FILE_PATH = './trained_model/model_full.pth'
OPT_FILE_PATH = './trained_model/opt.pth'

epochs = 10
batch_size = 32
max_grad_norm = 5
lr = 0.0005
max_patience = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


tag2idx: dict
with open(TAG_VOCAB_PATH, 'r') as f:
    tag2idx = json.load(f)
    tag2idx['B-<START>'] = tag2idx.pop('<START>')
    tag2idx['B-<END>'] = tag2idx.pop('<END>')
idx2tag = {i: tag for tag, i in tag2idx.items()}
vocab: dict
with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)


def train(model, optimizer, train_dataloader):
    model.train()

    report_every = 500
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

        if step % report_every == 0:
            print(f'{step}/{len(train_dataloader)} loss: {tr_loss/n_sentences}')

    print(f'Train Loss: {tr_loss/n_sentences}')


def val(model, val_dataloader):
    model.eval()

    val_loss, n_sentences = 0, 0

    pred_tags, true_tags = [], []
    # one epoch
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
        pred_tags.extend([tags for tags in tmp_pred_tags])
        true_tags.extend([tags for tags in tmp_true_tags])

    print(f'Avg Validation loss: {val_loss/n_sentences}')
    print(f'Validation F1: {f1_score(true_tags,pred_tags)}')
    print(classification_report(true_tags, pred_tags))
    print(
        f"Validation F1 STRICT: {f1_score(true_tags,pred_tags,mode='strict',scheme=IOB2)}")
    print(classification_report(true_tags, pred_tags, scheme=IOB2, mode='strict'))

    return val_loss/n_sentences


if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE_PATH, converters={
        'x_ids': eval, 'y_ids': eval}, index_col=0)
    input_ids, tag_ids = df['x_ids'].tolist(), df['y_ids'].tolist()

    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(
        input_ids, tag_ids, random_state=1234, test_size=0.1)

    train_data = LogDataset(tr_inputs, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size, collate_fn=collate_fn)

    val_data = LogDataset(val_inputs, val_tags)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(
        val_data, sampler=val_sampler, batch_size=batch_size, collate_fn=collate_fn)

    model = BiLSTMCRF(len(vocab), len(tag2idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, verbose=True)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(f'EPOCH: {epoch}/{epochs}')

        train(model, optimizer, train_dataloader)
        val_loss = val(model, val_dataloader)
        scheduler.step(val_loss)

        patience, decay = 0, 0
        if val_loss < best_val_loss:
            patience = 0
            torch.save(model.state_dict(), MODEL_SAVE_FILE_PATH)
        else:
            patience += 1
            if patience == max_patience:
                print('early stop')
                exit()

    pass
