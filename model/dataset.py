import torch
from preprocess.utils import len2mask, pad


class LogDataset(torch.utils.data.Dataset):
    def __init__(self, x_ids, y_ids):
        self.x_ids = x_ids
        self.y_ids = y_ids

        if len(x_ids) != len(self.y_ids):
            raise ValueError("lengths are not equal")

    def __len__(self):
        return len(self.x_ids)

    def __getitem__(self, idx):
        return self.x_ids[idx], self.y_ids[idx]


def collate_fn(batch, return_tensor=True, x_pad_idx=0, y_pad_idx=0):
    b_input_ids, b_tag_ids = [t[0] for t in batch], [t[1] for t in batch]

    seq_lens = [len(inputs) for inputs in b_input_ids]
    b_masks = len2mask(seq_lens)

    # pad seq in bath to same length
    max_len = max(seq_lens)
    b_input_ids = [pad(input_ids, x_pad_idx, max_len)
                   for input_ids in b_input_ids]
    b_tag_ids = [pad(input_labels, y_pad_idx, max_len)
                 for input_labels in b_tag_ids]

    if return_tensor:
        return torch.LongTensor(b_input_ids), torch.LongTensor(b_tag_ids), torch.LongTensor(b_masks)
    else:
        return b_input_ids, b_tag_ids, b_masks
