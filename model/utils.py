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
