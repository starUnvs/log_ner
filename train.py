import pandas as pd
from transformers import BertTokenizer, GPT2Tokenizer, BertForTokenClassification
from preprocess.utils import *
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
from seqeval.metrics import classification_report, accuracy_score, f1_score


MAX_LENGTH = 256

df = pd.read_csv(
    './data.csv', converters={'header_anno': eval, 'msg_anno': eval}, index_col=0)[:10]
header_anno = df.header_anno.tolist()
msg_anno = df.msg_anno.tolist()
log_anno = [h+m for h, m in zip(header_anno, msg_anno)]

log = []
tag = []
for anno in log_anno:
    tmp_words = [word for word, tag in anno]
    tmp_tags = [tag for word, tag in anno]
    log.append(tmp_words)
    tag.append(tmp_tags)

#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer =BertTokenizer.from_pretrained('bert-base-cased')
subword_log, subword_tag = subword_tokenize(log, tag, tokenizer)

tag_values = ['<DATE>', '<TIME>', '<LVL>', '<CLS>', '<FUNC>',
              '<HOST>', '<PATH>', '<URL>', '<O>', '<PAD>', 'X']
tag2idx = {t: i for i, t in enumerate(tag_values)}
idx2tag = {i: t for t, i in tag2idx.items()}

input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in subword_log]
tag_ids = []
for x in subword_tag:
    tag_ids.append([tag2idx[t] for t in x])

input_ids = [pad(x, 0, MAX_LENGTH) for x in input_ids]
tag_ids = [pad(x, tag2idx['<PAD>'], MAX_LENGTH) for x in tag_ids]

attention_masks = [[int(i > 0) for i in ii] for ii in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks = train_test_split(
    input_ids, tag_ids, attention_masks, random_state=1234, test_size=0.1)

device = torch.device("cuda")

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=32)

model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(tag2idx))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


epochs = 2
max_grad_norm = 1.0

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        outputs = model(input_ids=b_input_ids,
                             attention_mask=b_input_mask, labels=b_labels)
        loss=outputs.loss
        score=outputs.logits
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                                          attention_mask=b_input_mask, labels=b_labels)
            tmp_eval_loss,logits=outputs.loss,outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tag_values[p_i] for p in predictions for p_i in p]
    valid_tags = [tag_values[l_ii]
                  for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

pass
