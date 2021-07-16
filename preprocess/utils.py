def pad(seq, content, width):
    return seq+[content]*(width-len(seq))


def len2mask(seq_lens):
    max_len = max(seq_lens)
    mask = [[1]*seq_len+[0]*(max_len-seq_len) for seq_len in seq_lens]

    return mask


if __name__ == '__main__':
    mask = len2mask([2, 4])

    pass
