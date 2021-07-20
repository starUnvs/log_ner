import json

import pandas as pd
from tqdm import tqdm

from preprocess.tokenizer import BaseTokenizer

logfile = './datasets/spark/spark_ds.csv'
idsfile = './inputs/spark.csv'
vocab_file = './vocab/vocab_full.json'


def create_inputs(nwords, ntags, tokenizer, tag2idx):
    x_inputs, y_inputs = [], []
    for words, tags in tqdm(zip(nwords, ntags)):
        x_ids, y_ids = [tokenizer.vocab['<START>']], [tag2idx['<START>']]

        for word, tag in zip(words, tags):
            tokens = tokenizer.tokenize(word)
            x_ids.extend(tokenizer.convert_tokens_to_ids(tokens))

            y_ids.append(tag2idx[tag])

            if len(tokens) == 1:
                continue

            if tag[0] == 'B' or tag[0] == 'I':
                post_tag = 'I'+tag[1:]
            elif tag == 'O':
                post_tag = 'X'
            else:
                raise ValueError("tag value invalid")

            y_ids.extend([tag2idx[post_tag]]*(len(tokens)-1))
        x_ids.append(tokenizer.vocab['<END>'])
        y_ids.append(tag2idx['<END>'])

        new_tokens = tokenizer.convert_ids_to_tokens(x_ids)
        new_log = ''.join(new_tokens)

        x_inputs.append(x_ids)
        y_inputs.append(y_ids)

    return x_inputs, y_inputs


if __name__ == '__main__':
    tag2idx = json.load(open('./vocab/tag_vocab.json'))
    # tag2idx['X'] = 100
    tokenizer = BaseTokenizer(vocab_file=vocab_file)

    df = pd.read_csv(logfile, converters={'tokens': eval, 'tags': eval})
    nwords, ntags = df['tokens'].to_list(), df['tags'].to_list()

    x_ids, y_ids = create_inputs(nwords, ntags, tokenizer, tag2idx)
    pd.DataFrame({'x_ids': x_ids, 'y_ids': y_ids}).to_csv(idsfile)
    print('finished')

    pass
