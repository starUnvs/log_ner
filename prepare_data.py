from preprocess.tokenizer import BaseTokenizer
import pandas as pd
from tqdm import tqdm

logfile = './datasets/apache/apache_ds.csv'
idsfile = './datasets/apache/input_ids.csv'
vocab_file = './vocab_100k.json'


def create_inputs(nwords, ntags, tokenizer, tag2idx):
    x_inputs, y_inputs = [], []
    for words, tags in tqdm(zip(nwords, ntags)):
        x_ids, y_ids = [], []

        for word, tag in zip(words, tags):
            tokens = tokenizer.tokenize(word)
            x_ids.extend(tokenizer.convert_tokens_to_ids(tokens))

            y_ids.append(tag2idx[tag])

            if len(tokens) == 1:
                continue

            if tag[0] == 'B' or tag[0] == 'I':
                post_tag = 'I'+tag[1:]
            elif tag[0] == 'O':
                post_tag = 'X'

            y_ids.extend([tag2idx[post_tag]]*(len(tokens)-1))
        
        new_tokens=tokenizer.convert_ids_to_tokens(x_ids)
        new_log=''.join(new_tokens)

        x_inputs.append(x_ids)
        y_inputs.append(y_ids)

    return x_inputs, y_inputs


if __name__ == '__main__':
    tag_values = ['B-DATE', 'B-TIME', 'B-LVL', 'B-FUNC', 'B-CLS', 'B-HOST', 'B-IP', 'B-PATH',
                  'I-DATE', 'I-TIME', 'I-LVL', 'I-FUNC', 'I-CLS', 'I-HOST', 'I-IP', 'I-PATH', '<O>','X','<UNK>']
    tag2idx = {tag: i for i, tag in enumerate(tag_values)}
    tag2idx['X'] = 100
    tokenizer = BaseTokenizer(vocab_file=vocab_file)

    df = pd.read_csv(logfile, converters={'tokens': eval, 'tags': eval})
    nwords, ntags = df['tokens'].to_list(), df['tags'].to_list()

    x_ids, y_ids = create_inputs(nwords, ntags, tokenizer, tag2idx)
    pd.DataFrame({'x_ids':x_ids,'y_ids':y_ids}).to_csv(idsfile)

    pass
