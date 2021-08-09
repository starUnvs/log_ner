import pandas as pd


def pad(seq, pad_content, width):
    return seq+[pad_content]*(width-len(seq))


def len2mask(seq_lens):
    max_len = max(seq_lens)
    mask = [[1]*seq_len+[0]*(max_len-seq_len) for seq_len in seq_lens]

    return mask


def load_data(data_file_path):
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
    return nwords, ntags


def align_two_seq(seq1, seq2):
    for i in range(len(seq1)):
        width = max(len(seq1[i]), len(seq2[i]))
        seq1[i] = str.ljust(seq1[i], width)
        seq2[i] = str.ljust(seq2[i], width)
    return seq1, seq2


def _get_next_tag(tag):
    if tag[0] == 'B' or tag[0] == 'I':
        next_tag = 'I'+tag[1:]
    elif tag[0] == 'O' or tag[0] == 'X':
        next_tag = 'X'
    else:
        raise ValueError('wrong tag')
    return next_tag


def merge(words, tags):
    slices = []
    start = 0
    next_tag = _get_next_tag(tags[start])
    for i, tag in enumerate(tags[1:], 1):
        if tag != next_tag:
            slices.append((start, i))
            start = i
            next_tag = _get_next_tag(tag)
    slices.append((start, len(tags)))

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
