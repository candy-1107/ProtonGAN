import numpy as np
import torch

PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# 独热编码
def one_hot_encode(seq, max_seq_length):
    encode = np.zeros((max_seq_length, len(PROTEIN_ALPHABET)))
    for i, base in enumerate(seq):
        index = PROTEIN_ALPHABET.index(base)
        encode[i, index] = 1
    return torch.tensor(encode, dtype=torch.float32)

# 将生成的独热编码序列转换为氨基酸序列
def decode_sequences(generated_sequences):
    protein_sequences = []
    for seq in generated_sequences:
        seq = seq.cpu().detach().numpy()
        new_seq = ""
        for k in seq:
            index = np.argmax(k)
            new_seq += PROTEIN_ALPHABET[index]
        protein_sequences.append(new_seq)
    return protein_sequences