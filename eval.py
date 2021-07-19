import json

import pandas as pd
import torch
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model.bilstm_crf import BiLSTMCRF
from model.dataset import LogDataset, collate_fn

MODEL_PATH = './trained_model/model_full.pth'
TEST_DATA_PATH = './inputs/test.csv'
VOCAB_PATH = './vocab/vocab_full.json'
TAG_VOCAB_PATH = './vocab/tag_vocab.json'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, test_dataloader):
    model.eval()

    total_loss, n_sentences = 0, 0

    pred_tags, true_tags = [], []
    # one epoch
    for batch in tqdm(test_dataloader):
        b_input_ids, b_tag_ids, b_masks = batch

        # to tensor
        b_input_ids = b_input_ids.to(device)
        b_tag_ids = b_tag_ids.to(device)
        b_masks = b_masks.to(device)

        with torch.no_grad():
            loss = model(b_input_ids, b_tag_ids, b_masks)
            b_pred_ids = model.predict(b_input_ids, b_masks)

        # record loss
        total_loss += loss.item()
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

    print(f'Avg loss: {total_loss/n_sentences}')
    print(f'F1: {f1_score(true_tags,pred_tags)}')
    print(classification_report(true_tags, pred_tags))
    print(
        f"F1 STRICT: {f1_score(true_tags,pred_tags,mode='strict',scheme=IOB2)}")
    print(classification_report(true_tags, pred_tags, scheme=IOB2, mode='strict'))

    return total_loss/n_sentences


if __name__ == '__main__':
    tag2idx: dict
    with open(TAG_VOCAB_PATH, 'r') as f:
        tag2idx = json.load(f)
        tag2idx['B-<START>'] = tag2idx.pop('<START>')
        tag2idx['B-<END>'] = tag2idx.pop('<END>')
    idx2tag = {i: tag for tag, i in tag2idx.items()}

    vocab: dict
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    model = BiLSTMCRF(len(vocab), len(tag2idx)).to(device)
    check_point = torch.load(MODEL_PATH)
    model.load_state_dict(check_point)

    df = pd.read_csv(TEST_DATA_PATH, converters={
        'x_ids': eval, 'y_ids': eval}, index_col=0)
    input_ids, tag_ids = df['x_ids'].tolist(), df['y_ids'].tolist()

    test_ds = LogDataset(input_ids, tag_ids)
    test_dl = DataLoader(test_ds, batch_size=32, collate_fn=collate_fn)

    test(model, test_dl)
