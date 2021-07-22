import argparse
import json

import tensorboardX as tb
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from model.bilstm_crf import BiLSTMCRF
from model.dataset import LogDataset, collate_fn
from preprocess.tokenizer import BaseTokenizer
from preprocess.utils import load_data


def train(model, optimizer, train_dataloader, val_dataloader, writer, epoch, report_every=30):
    model.train()

    tr_loss, n_sentences = 0, 0
    best_f1_strict = 0.0

    for step, batch in enumerate(tqdm(train_dataloader), epoch*len(train_dataloader)):
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
            parameters=model.parameters(), max_norm=5)
        optimizer.step()

        # track train loss
        tr_loss += loss.item()
        n_sentences += b_input_ids.shape[0]

        writer.add_scalar('train_loss', loss.item()/b_input_ids.shape[0], step)

        if step % report_every == 0:
            test_loss, f1, f1_strict = model.test(
                val_dataloader, verbose=False)
            model.train()

            writer.add_scalar('test_loss', test_loss, step)
            writer.add_scalar('f1', f1, step)
            writer.add_scalar('f1_strict', f1_strict, step)

            # scheduler.step(test_loss)

            if f1_strict > best_f1_strict:
                torch.save(model, writer.logdir+'/model.pth')

            best_f1_strict = tr_loss/n_sentences
            tr_loss = 0
            n_sentences = 0


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-esize', '--embedding_size', type=int, default=128)
    parser.add_argument('-hsize', '--hidden_size', type=int, default=256)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0025)
    parser.add_argument('-bsize', '--batch_size', type=int, default=64)
    parser.add_argument('-e', '--epoch', type=int, default=5)
    parser.add_argument('-p', '--model_dir', type=str,
                        default='./run/')
    parser.add_argument('-v2i', '--vocab2idx_path', type=str,
                        default='./vocab/vocab.json')
    parser.add_argument('-t2i', '--tag2idx_path', type=str,
                        default='./vocab/tag_vocab.json')
    parser.add_argument('-train', '--train_path', type=str,
                        default='./inputs/train.csv')
    parser.add_argument('-test', '--test_path', type=str,
                        default='./inputs/spark.csv')

    args = parser.parse_args()

    train_data = LogDataset(*load_data(args.train_path))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    val_data = LogDataset(*load_data(args.test_path))
    val_sampler = RandomSampler(val_data, replacement=True, num_samples=1000)
    val_dataloader = DataLoader(
        val_data, sampler=val_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    with open(args.vocab2idx_path, 'r') as f:
        vocab2idx = json.load(f)
    with open(args.tag2idx_path, 'r') as f:
        tag2idx = json.load(f)

    model = BiLSTMCRF(embed_size=args.embedding_size, hidden_size=args.hidden_size,
                      tokenizer=BaseTokenizer(), vocab2idx=vocab2idx, tag2idx=tag2idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, verbose=True, patience=10)

    result_dir = args.model_dir + \
        f'lr{args.learning_rate}_batch{args.batch_size}_e{args.embedding_size}_h{args.hidden_size}'
    writer = tb.SummaryWriter(result_dir)

    for epoch in range(args.epoch):
        print(f'EPOCH: {epoch+1}/{args.epoch}')

        train(model, optimizer, train_dataloader,
              val_dataloader, writer, epoch)

    pass
