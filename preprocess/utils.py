import pandas as pd


def pad(seq, content, width):
    return seq+[content]*(width-len(seq))


def len2mask(seq_lens):
    max_len = max(seq_lens)
    mask = [[1]*seq_len+[0]*(max_len-seq_len) for seq_len in seq_lens]

    return mask


def load_data(data_file_path):
    df = pd.read_csv(data_file_path, converters={
        'x_ids': eval, 'y_ids': eval}, index_col=0)
    x_ids, y_ids = df['x_ids'].tolist(), df['y_ids'].tolist()
    return x_ids, y_ids


def align_two_seq(seq1, seq2):
    for i in range(len(seq1)):
        width = max(len(seq1[i]), len(seq2[i]))+2
        seq1[i] = str.center(seq1[i], width)
        seq2[i] = str.center(seq2[i], width)
    return seq1, seq2


if __name__ == '__main__':
    mask = len2mask([2, 4])

    pass
