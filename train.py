import json

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from model.bilstm_crf import BiLSTMCRF
from model.dataset import LogDataset, collate_fn
from preprocess.utils import load_data

DATA_FILE_PATH = './inputs/train.csv'
VOCAB_PATH = './vocab/vocab_full.json'
TAG_VOCAB_PATH = './vocab/tag_vocab.json'
MODEL_SAVE_FILE_PATH = './trained_model/model_128.pth'
OPT_FILE_PATH = './trained_model/opt.pth'

epochs = 10
batch_size = 64
max_grad_norm = 5
lr = 0.01
max_patience = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

embedding_size = 128
hidden_size = 128


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


if __name__ == '__main__':
    x_ids, y_ids = load_data(DATA_FILE_PATH)

    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(
        x_ids, y_ids, random_state=1234, test_size=0.1)

    train_data = LogDataset(tr_inputs, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size, collate_fn=collate_fn)

    val_data = LogDataset(val_inputs, val_tags)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(
        val_data, sampler=val_sampler, batch_size=batch_size, collate_fn=collate_fn)

    model = BiLSTMCRF(len(vocab), len(tag2idx),
                      embed_size=embedding_size, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, verbose=True)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(f'EPOCH: {epoch}/{epochs}')

        train(model, optimizer, train_dataloader)
        val_loss, _ = model.test(val_dataloader, device, idx2tag)
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
