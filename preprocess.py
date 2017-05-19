# From https://github.com/nschuc/alternating-reader-tf/blob/master/load_data.py, some modifications are made

import json
import os
import numpy as np
import torch
from functools import reduce
import itertools
import time

# parallel processing
from joblib import Parallel, delayed

import aoareader.Constants
from aoareader.Dict import Dict as Vocabulary

from nltk.tokenize import word_tokenize

data_path = 'data/'
data_filenames = {
        'train': 'train.txt',
        'valid': 'dev.txt'
        }
preprocessed = os.path.join(data_path, 'preprocessed.pt')
vocab_file = os.path.join(data_path, 'vocab.json')


def tokenize(sentence):
    return [s.strip() for s in word_tokenize(sentence) if s.strip()]


def parse_stories(lines):
    stories = []
    story = []
    for line in lines:
        line = line.strip()
        if not line:
            story = []
        else:
            _, line = line.split(' ', 1)
            if line:
                if '\t' in line:  # query line
                    q, a, _, answers = line.split('\t')
                    q = tokenize(q)

                    # use the first 10
                    candidates = answers.split('|')[:10]
                    stories.append((story, q, a, candidates))
                else:
                    story.append(tokenize(line))
    return stories


def get_stories(story_lines):
    stories = parse_stories(story_lines)
    flatten = lambda story: reduce(lambda x, y: x + y, story)
    stories = [(flatten(story), q, a, candidates) for story, q, a, candidates in stories]
    return stories


def vectorize_stories(stories, vocab : Vocabulary):
    X = []
    Q = []
    C = []
    A = []

    for s, q, a, c in stories:
        x = vocab.convert2idx(s)
        xq = vocab.convert2idx(q)
        xc = vocab.convert2idx(c)
        X.append(x)
        Q.append(xq)
        C.append(xc)
        A.append(vocab.getIdx(a))

    X = X
    Q = Q
    C = C
    A = torch.LongTensor(A)
    return X, Q, A, C


def build_dict(stories):
    if os.path.isfile(vocab_file):
        with open(vocab_file, "r") as vf:
            word2idx = json.load(vf)
    else:

        vocab = sorted(set(itertools.chain(*(story + q + [answer] + candidates
                                             for story, q, answer, candidates in stories))))
        vocab_size = len(vocab) + 2     # pad, unk
        print('Vocab size:', vocab_size)
        word2idx = dict((w, i + 2) for i,w in enumerate(vocab))
        word2idx[aoareader.Constants.UNK_WORD] = 1
        word2idx[aoareader.Constants.PAD_WORD] = 0

        with open(vocab_file, "w") as vf:
            json.dump(word2idx, vf)

    return Vocabulary(word2idx)


def main():

    print('Preparing process dataset ...')
    train_filename = os.path.join(data_path, data_filenames['train'])
    valid_filename = os.path.join(data_path, data_filenames['valid'])
    with open(train_filename, 'r') as tf, open(valid_filename, 'r') as vf:
        tlines = tf.readlines()
        vlines = vf.readlines()
        train_stories, valid_stories = Parallel(n_jobs=2)(delayed(get_stories)(story_lines)
                                                          for story_lines in [tlines, vlines])

    print('Preparing build dictionary ...')
    vocab_dict = build_dict(train_stories + valid_stories)

    print('Preparing training and validation ...')
    train = {}
    valid = {}

    train_data, valid_data = Parallel(n_jobs=2)(delayed(vectorize_stories)(stories, vocab_dict)
                                                for stories in [train_stories, valid_stories])
    train['documents'], train['querys'], train['answers'], train['candidates'] = train_data
    valid['documents'], valid['querys'], valid['answers'], valid['candidates'] = valid_data

    print('Saving data to \'' + preprocessed + '\'...')
    save_data = {'dict': vocab_dict,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, preprocessed)

if __name__ == '__main__':
    main()
