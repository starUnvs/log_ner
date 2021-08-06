import argparse
import json

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.dataset import LogNERDataset, collate
from model.model import BiLSTMLogNER, CNNLogNER
from model.tokenizer import BaseTokenizer
from model.utils import load_data


def train(model, optimizer, train_dataloader, writer, epoch):
    model.train()

    for step, batch in enumerate(tqdm(train_dataloader), (epoch-1)*len(train_dataloader)+1):
        b_word_ids, b_masks, b_char_ids, b_tag_ids = batch

        # to tensor
        b_word_ids = b_word_ids.to(device)
        b_masks = b_masks.to(device)
        b_char_ids = b_char_ids.to(device)
        b_tag_ids = b_tag_ids.to(device)

        loss = model(b_word_ids, b_masks, b_char_ids, b_tag_ids)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=5.0)
        optimizer.step()

        writer.add_scalar('loss/train', loss.item()/b_word_ids.shape[0], step)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-clevel', '--char_level', type=bool, default=True)
    parser.add_argument('-esize', '--embedding_size', type=int, default=128)
    parser.add_argument('-cesize', '--char_embedding_size',
                        type=int, default=25)
    parser.add_argument('-cfsize', '--char_feature_size',
                        type=int, default=50)
    parser.add_argument('-hsize', '--hidden_size', type=int, default=256)
    parser.add_argument('-csize', '--conv_size', type=int, default=256)
    parser.add_argument('-fcsize', '--fc_size', type=int, default=1024)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-bsize', '--batch_size', type=int, default=64)
    parser.add_argument('-e', '--epoch', type=int, default=15)
    parser.add_argument('-p', '--model_dir', type=str,
                        default='./run/')
    parser.add_argument('-m', '--model', type=str, default='bilstm')
    parser.add_argument('-v2i', '--word2idx_path', type=str,
                        default='./vocab/word_vocab.json')
    parser.add_argument('-c2i', '--char2idx_path', type=str,
                        default='./vocab/char_vocab.json')
    parser.add_argument('-t2i', '--tag2idx_path', type=str,
                        default='./vocab/tag_vocab.json')
    parser.add_argument('-train', '--train_path', type=str,
                        default='./datasets/train_without_hadoop.csv')
    parser.add_argument('-test', '--test_path', type=str,
                        default='./datasets/test_hadoop.csv')

    args = parser.parse_args()

    with open(args.word2idx_path, 'r') as f:
        word2idx = json.load(f)
    with open(args.tag2idx_path, 'r') as f:
        tag2idx = json.load(f)
    with open(args.char2idx_path, 'r') as f:
        char2idx = json.load(f)
    tokenizer = BaseTokenizer()

    nwords, ntags = load_data(args.train_path)
    tr_x, test_x, tr_y, test_y = train_test_split(
        nwords, ntags, test_size=0.2, random_state=1234, shuffle=True)

    train_data = LogNERDataset(
        tr_x, tr_y, tokenizer, word2idx, char2idx, tag2idx)
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, collate_fn=collate)

    val_data = LogNERDataset(test_x, test_y, tokenizer,
                             word2idx, char2idx, tag2idx)
    val_dataloader = DataLoader(
        val_data, batch_size=args.batch_size, collate_fn=collate)

    if args.model == 'bilstm':
        model = BiLSTMLogNER(char_level=args.char_level,
                             word_embed_size=args.embedding_size,
                             char_embed_size=args.char_embedding_size,
                             char_feat_size=args.char_feature_size,
                             hidden_size=args.hidden_size,
                             tokenizer=tokenizer,
                             word2idx=word2idx,
                             char2idx=char2idx,
                             tag2idx=tag2idx
                             )
        result_dir = args.model_dir + \
            f"{args.model}_lr{args.learning_rate}_we{args.embedding_size}_ce{args.char_embedding_size}_cf{args.char_feature_size}_h{args.hidden_size}_data({args.train_path.split('/')[-1]})/"
    elif args.model == 'cnn':
        model = CNNLogNER(char_level=args.char_level,
                          word_embed_size=args.embedding_size,
                          char_embed_size=args.char_embedding_size,
                          char_feat_size=args.char_feature_size,
                          conv_size=args.conv_size,
                          fc_size=args.fc_size,
                          tokenizer=tokenizer,
                          word2idx=word2idx,
                          char2idx=char2idx,
                          tag2idx=tag2idx
                          )
        result_dir = args.model_dir + \
            f"{args.model}_lr{args.learning_rate}_we{args.embedding_size}_ce{args.char_embedding_size}_cf{args.char_feature_size}_conv{args.conv_size}_fc{args.fc_size}_data({args.train_path.split('/')[-1]})/"
    else:
        raise ValueError("No module error")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, verbose=True, patience=3, threshold=1e-4, threshold_mode='abs')

    writer = SummaryWriter(result_dir)

    test_x, test_y = load_data(args.test_path)
    test_dataloader = DataLoader(
        LogNERDataset(test_x, test_y, tokenizer, word2idx, char2idx, tag2idx), batch_size=args.batch_size, collate_fn=collate)

    best_val_f1 = float('-inf')
    for epoch in range(1, args.epoch+1):
        print(f'EPOCH: {epoch}/{args.epoch}')
        train(model, optimizer, train_dataloader, writer, epoch)

        loss, f1, f1_strict = model.test(
            val_dataloader, verbose=False)

        writer.add_scalar('loss/val', loss, epoch)
        writer.add_scalars('metric/val', {
            'val_f1': f1, 'val_f1strict': f1_strict}, epoch)

        scheduler.step(loss)

        if f1 > best_val_f1:
            best_val_f1 = f1
            print('Saving model...')
            torch.save(model, writer.log_dir+'model.pth')

        loss, f1, f1_strict = model.test(test_dataloader)
        writer.add_scalars(
            'metric/test', {'f1': f1, 'f1_strict': f1_strict}, epoch)

    pass
