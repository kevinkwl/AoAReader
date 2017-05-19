import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F
from aoareader import Constants


def sort_batch(data, seq_len):
    batch_size = data.size(0)
    sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)
    sorted_data = data[sorted_idx]
    _, reverse_idx = torch.sort(sorted_idx, dim=0, descending=False)
    return sorted_data, sorted_seq_len, reverse_idx


# From https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637
def softmax(input, axis=1):
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size) - 1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size) - 1)


class AoAReader(nn.Module):

    def __init__(self, vocab_dict, dropout_rate, embed_dim, hidden_dim, bidirectional=True):
        super(AoAReader, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(vocab_dict.size(),
                                      self.embed_dim,
                                      padding_idx=Constants.PAD)

        input_size = self.embed_dim
        self.gru = nn.GRU(input_size, hidden_size=self.hidden_dim, dropout=dropout_rate,
                          bidirectional=bidirectional, batch_first=True)

    def forward(self, docs_input, docs_len, querys_input, querys_len):
        s_docs, s_docs_len, reverse_docs_idx = sort_batch(docs_input, docs_len)
        s_querys, s_querys_len, reverse_querys_idx = sort_batch(querys_input, querys_len)

        docs_embedding = pack(self.embedding(s_docs), list(s_docs_len), batch_first=True)
        querys_embedding = pack(self.embedding(s_querys), list(s_querys_len), batch_first=True)

        # encode
        docs_outputs, _ = self.gru(docs_embedding, None)
        querys_outputs, _ = self.gru(querys_embedding, None)

        # unpack
        docs_outputs, _ = unpack(docs_outputs, batch_first=True)
        querys_outputs, _ = unpack(querys_outputs, batch_first=True)

        docs_outputs = docs_outputs[reverse_docs_idx]
        querys_outputs = querys_outputs[reverse_querys_idx]


        # transpose query for pair-wise dot product
        dos = docs_outputs
        qos = torch.transpose(querys_outputs, 1, 2)

        # pair-wise matching score
        M = torch.bmm(dos, qos)

        # query-document attention
        alpha = softmax(M, axis=1)
        beta = softmax(M, axis=2)

        print(docs_outputs.size())
        print(querys_outputs.size())
        print(docs_outputs)
        print(querys_outputs)
        return beta






