import json

import torch
from torch.utils.data.dataloader import DataLoader

from model.bilstm_crf import BiLSTMCRF
from model.dataset import LogDataset, collate_fn
from preprocess.utils import load_data

MODEL_PATH = './trained_model/model_full.pth'
TEST_DATA_PATH = './inputs/test_full.csv'
VOCAB_PATH = './vocab/vocab_full.json'
TAG_VOCAB_PATH = './vocab/tag_vocab.json'


if __name__ == '__main__':
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
    vocab2idx = {key: i for i, key in enumerate(vocab.keys())}
    idx2vocab = {i: key for key, i in vocab2idx.items()}

    model = BiLSTMCRF(len(vocab), len(tag2idx),
                      embed_size=64, hidden_size=64).to(device)
    check_point = torch.load(MODEL_PATH)
    model.load_state_dict(check_point)

    x_ids, y_ids = load_data(TEST_DATA_PATH)
    test_dl = DataLoader(LogDataset(x_ids, y_ids),
                         batch_size=32, collate_fn=collate_fn)

    model.test(test_dl, idx2vocab, idx2tag, device, True, './result.txt')
