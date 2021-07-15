import torch


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
