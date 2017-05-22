from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import aoareader


def create_mask(seq_lens):
    mask = torch.zeros(len(seq_lens), torch.max(seq_lens))
    for i, seq_len in enumerate(seq_lens):
        mask[i][:seq_len] = 1

    return mask.float()

class Dataset(object):

    def __init__(self, data: dict, batch_size, cuda, volatile=False):
        self.documents = data['documents']
        self.querys = data['querys']
        self.candidates = data['candidates']
        self.answers = data.get('answers', None)

        # check if dimensions match
        assert len(self.documents) == len(self.querys) == len(self.candidates)

        if self.answers is not None:
            assert len(self.querys) == len(self.answers)

        self.cuda = cuda

        self.batch_size = batch_size
        self.numBatches = math.ceil(len(self.querys)/batch_size)
        self.volatile = volatile

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(aoareader.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)

        documents, doc_lengths = self._batchify(
            self.documents[index*self.batch_size:(index+1)*self.batch_size],
            align_right=False, include_lengths=True)

        querys, q_lengths = self._batchify(
            self.querys[index*self.batch_size:(index+1)*self.batch_size],
            align_right=False, include_lengths=True)

        candidates = self._batchify(
            self.candidates[index*self.batch_size:(index+1)*self.batch_size],
            align_right=False, include_lengths=False)

        if self.answers is not None:
            answers = torch.LongTensor(self.answers[index*self.batch_size:(index+1)*self.batch_size])
        else:
            answers = None

        def wrap(b: torch.LongTensor):
            if b is None:
                return b
            if len(b.size()) > 1:
                b = torch.stack(b, 0)
            b = b.contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile, requires_grad=False)
            return b

        doc_lengths = torch.LongTensor(doc_lengths)
        doc_mask = create_mask(doc_lengths)
        q_lengths = torch.LongTensor(q_lengths)
        q_mask = create_mask(q_lengths)

        return (wrap(documents), wrap(doc_lengths), wrap(doc_mask)), (wrap(querys), wrap(q_lengths), wrap(q_mask)), wrap(answers), wrap(candidates)

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.documents, self.querys, self.candidates, self.answers))
        self.documents, self.querys, self.candidates, self.answers = zip(*[data[i] for i in torch.randperm(len(data))])