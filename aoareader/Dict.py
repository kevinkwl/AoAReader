from aoareader import Constants
import torch

class Dict:

    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {idx: word for word, idx in word2idx.items()}

    def getIdx(self, word):
        return self.word2idx.get(word, Constants.UNK)

    def getWord(self, idx):
        return self.idx2word.get(idx, Constants.UNK_WORD)

    def convert2idx(self, words):
        vec = [self.getIdx(word) for word in words]

        return torch.LongTensor(vec)

    def convert2word(self, idxs):
        vec = [self.getWord(idx) for idx in idxs]
        return vec

    def size(self):
        return len(self.idx2word)

