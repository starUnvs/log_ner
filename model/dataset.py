import torch


class LogDataset(torch.utils.data.Dataset):
    def __init__(self, subword_logs, subword_tags):
        self.subword_logs = subword_logs
        self.subword_tags = subword_tags

        if len(subword_logs) != len(self.subword_tags):
            raise ValueError("lengths are not equal")

    def __len__(self):
        return len(self.subword_logs)

    def __getitem__(self, idx):
        return self.subword_logs[idx], self.subword_tags[idx]
