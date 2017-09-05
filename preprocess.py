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

from sys import argv

data_path = 'data/'
data_filenames = {
        'train': 'train.txt',
        'valid': 'dev.txt',
        'test': 'test.txt'
        }
vocab_file = os.path.join(data_path, 'vocab.json')
dict_file = os.path.join(data_path, 'dict.pt')

def tokenize(sentence):
    return [s.strip().lower() for s in word_tokenize(sentence) if s.strip()]


def parse_stories(lines, with_answer=True):
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
                    answer = ''
                    if with_answer:
                        q, answer, _, candidates = line.split('\t')
                        answer = answer.lower()
                    else:
                        q, _, candidates = line.split('\t')
                    q = tokenize(q)

                    # use the first 10
                    candidates = [cand.lower() for cand in candidates.split('|')[:10]]
                    stories.append((story, q, answer, candidates))
                else:
                    story.append(tokenize(line))
    return stories


def get_stories(story_lines, with_answer=True):
    stories = parse_stories(story_lines, with_answer=with_answer)
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
    test_filename = os.path.join(data_path, data_filenames['test'])


    with open(train_filename, 'r') as tf, open(valid_filename, 'r') as vf, open(test_filename, 'r') as tef:
        tlines = tf.readlines()
        vlines = vf.readlines()
        telines = tef.readlines()
        train_stories, valid_stories, test_stories = Parallel(n_jobs=2)(delayed(get_stories)(story_lines)
                                                          for story_lines in [tlines, vlines, telines])


    print('Preparing build dictionary ...')
    vocab_dict = build_dict(train_stories + valid_stories + test_stories)

    print('Preparing training, validation, testing ...')
    train = {}
    valid = {}
    test = {}

    train_data, valid_data, test_data = Parallel(n_jobs=2)(delayed(vectorize_stories)(stories, vocab_dict)
                                                for stories in [train_stories, valid_stories, test_stories])
    train['documents'], train['querys'], train['answers'], train['candidates'] = train_data
    valid['documents'], valid['querys'], valid['answers'], valid['candidates'] = valid_data
    test['documents'], test['querys'], test['answers'], test['candidates'] = test_data


    print('Saving data to \'' + data_path + '\'...')
    torch.save(vocab_dict, dict_file)
    torch.save(train, train_filename + '.pt')
    torch.save(valid, valid_filename + '.pt')
    torch.save(test, test_filename + '.pt')

if __name__ == '__main__':
    main()
