import pandas as pd


def pad(seq, pad_content, width):
    return seq+[pad_content]*(width-len(seq))


def len2mask(seq_lens):
    max_len = max(seq_lens)
    mask = [[1]*seq_len+[0]*(max_len-seq_len) for seq_len in seq_lens]

    return mask


def load_data(data_file_path, tokenizer, vocab2idx, tag2idx):
    """load ner dataset from csv file

    Args:
        data_file_path (str): path of csv file

    Returns:
        tokens (list[list[str]]): a list of words, e.g: [['hello','world'],['hello','guys']]
        tags   (list[list[str]]): same as above
    """
    df = pd.read_csv(data_file_path, converters={
        'tokens': eval, 'tags': eval}, index_col=0)
    nwords, ntags = df['tokens'].tolist(), df['tags'].tolist()
    x_ids, y_ids = create_inputs(nwords, ntags, tokenizer, vocab2idx, tag2idx)
    return x_ids, y_ids


def align_two_seq(seq1, seq2):
    for i in range(len(seq1)):
        width = max(len(seq1[i]), len(seq2[i]))
        seq1[i] = str.ljust(seq1[i], width)
        seq2[i] = str.ljust(seq2[i], width)
    return seq1, seq2


def create_inputs(nwords, ntags, tokenizer, vocab2idx, tag2idx, prefix='B-<START>', suffix='B-<END>'):
    unk_idx = vocab2idx['UNK']
    x_inputs, y_inputs = [], []
    for words, tags in zip(nwords, ntags):
        x_ids, y_ids = [],[]
        if prefix:
            x_ids.append(vocab2idx[prefix])
            y_ids.append(tag2idx[prefix])

        for word, tag in zip(words, tags):
            tokens = tokenizer.tokenize(word)
            x_ids.extend([vocab2idx.get(token, unk_idx) for token in tokens])
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
        
        if suffix:
            x_ids.append(vocab2idx[suffix])
            y_ids.append(tag2idx[suffix])

        x_inputs.append(x_ids)
        y_inputs.append(y_ids)

    return x_inputs, y_inputs

def merge(words, tags):
    slices = []
    start, end = 0, 1
    for i, tag in enumerate(tags[1:], 1):
        if tag[0] != 'I' and tag[0] != 'X':
            end = i
            slices.append((start, end))
            start = i
    slices.append((start, end+1))

    merged_words = []
    merged_tags = []
    for start, end in slices:
        merged_words.append(''.join(words[start:end]))
        first_tag = tags[start]
        if first_tag[0] == 'B' or first_tag[0] == 'I':
            merged_tags.append(first_tag[2:])
        else:
            merged_tags.append(first_tag)

    return merged_words, merged_tags


if __name__ == '__main__':
    mask = len2mask([2, 4])

    pass
