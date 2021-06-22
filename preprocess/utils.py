import pandas as pd

def subword_tokenize(logs, tags, tokenizer=None, width=512):
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

def pad(l, content, width):
    l.extend([content] * (width - len(l)))
    return l


if __name__ == '__main__':
    df = pd.read_csv(
        './data.csv', converters={'header_anno': eval, 'msg_anno': eval}, index_col=0)
