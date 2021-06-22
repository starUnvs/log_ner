import os
import pandas as pd
from transformers import BertTokenizer, GPT2Tokenizer, BertForTokenClassification
from preprocess.utils import *
from model.dataset import LogDataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from seqeval.metrics import classification_report, accuracy_score, f1_score

SUBWORD_FILE_PATH = './subword_data.csv'
DATA_FILE_PATH = './data'
SUBWORD_TOKENIZER = 'bert'
MAX_LENGTH = 256

epochs = 2
max_grad_norm = 1.0
batch_size = 4

#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

if not os.path.exists(SUBWORD_FILE_PATH):
    subword_tokenize2file(DATA_FILE_PATH, SUBWORD_FILE_PATH, tokenizer)
df = pd.read_csv(SUBWORD_FILE_PATH, converters={
                 'subword_log': eval, 'subword_tag': eval}, index_col=0)

subword_logs, subword_tags = df['subword_log'].tolist(
), df['subword_tag'].tolist()

tag_values = ['<DATE>', '<TIME>', '<LVL>', '<CLS>', '<FUNC>',
              '<HOST>', '<PATH>', '<URL>', '<O>', '<PAD>', 'X']
tag2idx = {t: i for i, t in enumerate(tag_values)}
idx2tag = {i: t for t, i in tag2idx.items()}

input_ids = [tokenizer.convert_tokens_to_ids(
    tokens) for tokens in subword_logs]
tag_ids = [[tag2idx[t] for t in subword_tag] for subword_tag in subword_tags]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(
    input_ids, tag_ids, random_state=1234, test_size=0.1)


def collate_fn(batch):
    b_input_ids, b_input_labels = [t[0] for t in batch], [t[1] for t in batch]

    seq_lens = [len(inputs) for inputs in b_input_ids]
    b_input_masks = len2mask(seq_lens)

    # pad seq in bath to same length
    max_len = max(seq_lens)
    b_input_ids = [pad(input_ids, 0, max_len) for input_ids in b_input_ids]
    b_input_labels = [pad(input_labels, tag2idx['<PAD>'], max_len)
                      for input_labels in b_input_labels]

    return b_input_ids, b_input_labels, b_input_masks


train_data = LogDataset(tr_inputs, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=batch_size, collate_fn=collate_fn)

val_data = LogDataset(val_inputs, val_tags)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(
    val_data, sampler=val_sampler, batch_size=batch_size, collate_fn=collate_fn)

model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(tag2idx))
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_labels, b_input_masks = batch

        # to tensor
        b_input_ids = torch.LongTensor(b_input_ids)
        b_input_labels = torch.LongTensor(b_input_labels)
        b_input_masks = torch.LongTensor(b_input_masks)

        # forward pass
        outputs = model(input_ids=b_input_ids,
                        attention_mask=b_input_masks, labels=b_input_labels)
        loss = outputs.loss
        score = outputs.logits
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

'''

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

'''

pass
