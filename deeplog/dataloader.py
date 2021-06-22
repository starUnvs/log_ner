# import numpy as np
# import keras
# from keras.preprocessing.sequence import pad_sequences
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms

# EMBEDDING_DIM = 768


# class DataGenerator(nn.Module):
#     def __init__(self, x, y):
#         'Initialization'
#         self.x = x
#         self.y = y
#         # self.batch_size = batch_size

#     def __len__(self):
#         'Denotes the number of batches'
#         return math.ceil(len(self.x) / self.batch_size)

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
#         # y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
#         x = self.x[index]
#         y = self.y[index]
#         x = pad_sequences(x, dtype='object', padding='post',
#                           value=np.zeros(EMBEDDING_DIM)).astype(np.float32)
#         # x = pad_sequence([torch.from_numpy(np.array(x)) for x in input_x], batch_first=True).float()
#         # print('x 类型：',type(x))
#         return x, y


import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

EMBEDDING_DIM = 768


class DataGenerator(Dataset):
    def __init__(self, x, y):
        'Initialization'
        self.x = x
        self.y = y
        # self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches'
        return math.ceil(len(self.x))

    def __getitem__(self, index):
        'Generate one batch of data'
        # x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        # y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        x = self.x[index]
        y = self.y[index]
        # x = pad_sequences(x, dtype='object', padding='post',
        #                   value=np.zeros(EMBEDDING_DIM)).astype(np.float32)

        #最大设置为40暂定
        num_tokens=x.shape[0]
        mix_num_boxes = min(int(num_tokens), 50)
        # mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((50, 768))
        mix_features_pad[:mix_num_boxes] = x[:mix_num_boxes]
        x=mix_features_pad
        # print(x.shape)
        # x = pad_sequence([torch.from_numpy(np.array(x)) for x in input_x], batch_first=True).float()
        # print('x 类型：',type(x))
        return x, y