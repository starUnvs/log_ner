import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Masking
from torch._C import device

from Attention import Attention
from dataloader import DataGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import math
import random
import copy
import warnings
from sklearn.metrics import precision_recall_fscore_support
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=int, default=0,
                    help="resume training of model (0/no, 1/yes)")
parser.add_argument("--load_path", type=str,
                    default='checkpoints/model-latest.pt', help="latest model path")
args = parser.parse_args()

warnings.filterwarnings('ignore')
## fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True

# hyper-parameters
EMBEDDING_DIM = 768
batch_size = 512
epochs = 40
rnn_units = 256
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load data
training_data = np.load(
    'preprocessed_data/training_data.npz', allow_pickle=True)
# load test data
testing_data = np.load(
    'preprocessed_data/testing_data.npz', allow_pickle=True)
x_train = testing_data['x']
y_train = testing_data['y']
x_test = training_data['x']
y_test = training_data['y']
del testing_data
del training_data


# # model
# model = Sequential()
# model.add(Masking(mask_value=0., input_shape=(None, EMBEDDING_DIM)))
# model.add(Bidirectional(LSTM(rnn_units, return_sequences=True)))
# # model.add()
# model.add(Attention(bias=False))
# model.add(Dense(2, activation='softmax'))
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop', metrics=['acc'])
# # print(model.summary())
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, dim=768, pad=50, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(Model, self).__init__()
        encoder_norm = nn.LayerNorm(dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            dim, nhead, dim_feedforward, dropout)
        self.trans_encder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=2, norm=encoder_norm)
        self.Lstm = nn.LSTM(dim, 768, 5)
        self.pos_encoder1 = PositionalEncoding(d_model=768)
        self.pos_encoder2 = LearnedPositionEncoding(d_model=768)
        self.fc1 = nn.Linear(dim * pad, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        B, _, _ = x.size()
        # x = x.permute(1,0,2) #LSTM
        # x,_ = self.Lstm(x)
        # pos_embed = self.pos_encoder1(x)
        # x = x + pos_embed
        # print('posencoder:',x.shape)
        x = self.trans_encder(x)  # mask默认None
        # print('encoder:',x.shape)
        # x = x.permute(1,0,2)

        x = x.contiguous().view(B, -1)

        # x=F.relu((x), inplace=True)
        x = self.fc1(x)

        x = self.fc2(x)

        return x


# train
train_generator = DataGenerator(x_train, y_train)
# test
test_generator = DataGenerator(x_test, y_test)
# train_loader
train_loader = torch.utils.data.DataLoader(
    train_generator, batch_size=batch_size, shuffle=True, num_workers=16)
# test_loader
test_loader = torch.utils.data.DataLoader(
    test_generator, batch_size=batch_size, shuffle=False, num_workers=16)

loss_all = []
acc_all = []
model = Model()
model = model.to(device)
device_ids = [0, 1]
model = torch.nn.DataParallel(model, device_ids=device_ids)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()

start_epoch = -1
if args.resume == 1:
    path_checkpoint = args.load_path
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print("resume training from epoch ", start_epoch)

print('Using device = ', device)


best_acc = np.inf
for epoch in range(start_epoch+1, epochs):
    train_loss = 0
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        x, y = data[0].to(device), data[1].to(device)
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # 关于调整学习率
    if epoch % 10 == 0 and epoch != 0:
        optimizer.param_groups[0]['lr'] *= 0.5
    # 手动调节了
    # if epoch == 3:01
    #     optimzer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
    # if epoch == 6:
    #     optimzer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    # if epoch == 9:
    #     optimzer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

    train_loss = train_loss / len(train_loader)
    print("epoch : {}/{}, loss = {:.6f}".format(epoch, epochs, train_loss))

    model.eval()
    n = 0.0
    acc = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            x, y = data[0].to(device), data[1].to(device)
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            out = model(x).cpu()
            if batch_idx == 0:
                y_pred = out
                y_true = y.cpu()
            else:
                y_pred = np.concatenate((y_pred, out), axis=0)
                y_true = np.concatenate((y_true, y.cpu()), axis=0)
            # acc += (out.argmax(1) == y.argmax(1)).float().sum().item()
            # n += len(y)
    # epoch_acc = acc / n
    # print("result : {}/{}, acc = {:.6f}".format(acc, n, epoch_acc))

    # calculate metrics
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    # print(y_true.shape)
    # print(y_pred.shape)
    report = precision_recall_fscore_support(y_true, y_pred, average='binary')
    with open(f'result_Transformer+encoder.txt', 'a', encoding='utf-8') as f:
        f.write('number of epochs:'+str(epoch))
        f.write('Number of testing data:'+str(x_test.shape[0])+'\n')
        f.write('Precision:'+str(report[0])+'\n')
        f.write('Recall:'+str(report[1])+'\n')
        f.write('F1 score:'+str(report[2])+'\n')
        f.write('\n')
        f.close()

    print(f'Number of testing data: {x_test.shape[0]}')
    print(f'Precision: {report[0]:.4f}')
    print(f'Recall: {report[1]:.4f}')
    print(f'F1 score: {report[2]:.4f}')

    # if epoch_acc < best_acc:
    #     best_acc = epoch_acc
    #     best_model = copy.deepcopy(model)
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch
    }
    ckpt_path = 'checkpoints/'
    torch.save(checkpoint, os.path.join(
        ckpt_path, f'model-{epoch}.pt'))
    torch.save(checkpoint, os.path.join(
        ckpt_path, f'model-latest.pt'))

    # torch.save(model.state_dict(
    # ), '/home/wangsen/ws/log_detection/Log-based-Anomaly-Detection-System/checkpoints/'+'my_model'+str(epoch)+'.h5')


# model.fit(train_generator, epochs=epochs)

# save model
# model.save('my_model.h5')
# best_model.save('my_model.h5')
