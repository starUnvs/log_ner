import random

import torch
from utils import len2mask, pad


class LogDataset(torch.utils.data.Dataset):
    def __init__(self, x_ids, y_ids, random_mask=False, ratio=0.2, tag_ids_be_replaced=None, repl_id=None, replace=False, max_repl_id=33604):
        if random_mask:
            for token_ids, tag_ids in zip(x_ids, y_ids):
                for i, (token_id, tag_id) in enumerate(zip(token_ids, tag_ids)):
                    if tag_id in tag_ids_be_replaced and random.random() < ratio:
                        if replace and random.random() < ratio/4:
                            token_ids[i] = int(random.random()*max_repl_id)
                        else:
                            token_ids[i] = repl_id

        self.x_ids = x_ids
        self.y_ids = y_ids

        if len(x_ids) != len(self.y_ids):
            raise ValueError("lengths are not equal")

    def __len__(self):
        return len(self.x_ids)

    def __getitem__(self, idx):
        return self.x_ids[idx], self.y_ids[idx]


def collate_fn(batch, return_tensor=True, x_pad_idx=0, y_pad_idx=0, max_len=200):
    b_input_ids, b_tag_ids = [t[0][:max_len]
                              for t in batch], [t[1][:max_len] for t in batch]

    seq_lens = [len(inputs) for inputs in b_input_ids]
    b_masks = len2mask(seq_lens)

    # pad seq in bath to same length
    b_max_len = max(seq_lens)
    b_input_ids = [pad(input_ids, x_pad_idx, b_max_len)
                   for input_ids in b_input_ids]
    b_tag_ids = [pad(input_labels, y_pad_idx, b_max_len)
                 for input_labels in b_tag_ids]

    if return_tensor:
        return torch.LongTensor(b_input_ids), torch.LongTensor(b_tag_ids), torch.LongTensor(b_masks)
    else:
        return b_input_ids, b_tag_ids, b_masks


if __name__ == '__main__':
    x = [[1, 1, 1, 2, 2, 2, 3, 3, 3]]
    y = [[1, 1, 1, 1, 0, 0, 0, 0, 0]]

    ds = LogDataset(x, y, random_mask=True, ratio=0.5,
                    tag_ids_be_replaced=[1], repl_id=9)

    pass
