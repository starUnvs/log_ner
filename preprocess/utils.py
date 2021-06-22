import pandas as pd
import numpy as np

def subword_tokenize(logs, tags, tokenizer=None):
    subword_logs = []
    subword_labels = []
    for log, tag in zip(logs, tags):
        temp_log = []
        temp_tags = []

        for i, word in enumerate(log):
            subword = tokenizer.tokenize(word)
            temp_log+=subword

            temp_tags.append(tag[i])
            temp_tags.extend('X'*(len(subword)-1))

        subword_logs.append(temp_log)
        subword_labels.append(temp_tags)

    return subword_logs, subword_labels

def subword_tokenize2file(inpath,outpath,tokenizer=None):
    df = pd.read_csv(
        inpath, converters={'header_anno': eval, 'msg_anno': eval}, index_col=0)
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

    subword_log, subword_tag = subword_tokenize(log, tag, tokenizer)

    pd.DataFrame({'subword_log':subword_log, 'subword_tag':subword_tag}).to_csv(outpath)

def pad(l, content, width):
    l.extend([content] * (width - len(l)))
    return l
    
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def len2mask(seq_lens):
    max_len=max(seq_lens)
    mask=[[1]*seq_len+[0]*(max_len-seq_len) for seq_len in seq_lens]

    return mask

if __name__ == '__main__':
    mask=len2mask([2,4])

    pass