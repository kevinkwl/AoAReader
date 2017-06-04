import aoareader as reader
import torch
import time
import argparse
import os

from preprocess import get_stories, vectorize_stories


parser = argparse.ArgumentParser(description="test.py")

parser.add_argument('-testdata', required=True,
                    help='Path to the test.txt, test.txt.pt will be used if exists.')

parser.add_argument('-dict', required=True,
                    help='Path to the dictionary file (Vocabulary class)')

parser.add_argument('-out', default='data/result.txt',
                    help='output file name.')

parser.add_argument('-model', required=True, help='path to the saved model.')


testopt = parser.parse_args()
print(testopt)

def load_testdata(testfile, vocab_dict):
    if os.path.exists(testfile + '.pt'):
        return torch.load(testfile + '.pt')
    else:
        testd = {}
        with open(testfile, 'r') as tf:
            tlines = tf.readlines()
            test_stories = get_stories(tlines, with_answer=False)
            testd['documents'], testd['querys'], _, testd['candidates'] = vectorize_stories(test_stories, vocab_dict)
        torch.save(testd, testfile + '.pt')
        return testd


def evalulate(model, data, vocab_dict):
    model.eval()
    answers = []
    for i in range(len(data)):
        (batch_docs, batch_docs_len, doc_mask), (batch_querys, batch_querys_len, query_mask), _ , candidates = data[i]

        pred_answers, _ = model(batch_docs, batch_docs_len, doc_mask,
                                    batch_querys, batch_querys_len, query_mask,
                                    candidates=candidates)

        answers.extend(pred_answers.data)


    model.train()
    return vocab_dict.convert2word(answers)

def main():
    print("Loading dict", testopt.dict)
    vocab_dict = torch.load(testopt.dict)

    print("Loading test data and vectorizing data")
    test_data = load_testdata(testopt.testdata, vocab_dict)

    print("Loading model from ", testopt.model)
    ckp = torch.load(testopt.model)

    opt = ckp['opt']
    model_state = ckp['model']

    test_dataset = reader.Dataset(test_data, opt.batch_size, opt.gpu, volatile=True)

    print(' * vocabulary size = %d' %
          (vocab_dict.size()))
    print(' * number of test samples. %d' %
          len(test_data['candidates']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    model = reader.AoAReader(vocab_dict, dropout_rate=opt.dropout, embed_dim=opt.embed_size, hidden_dim=opt.gru_size)
    # no way on CPU
    model.cuda()

    # load state
    model.load_state_dict(model_state)

    print('Evaluate on test data')
    answers = evalulate(model, test_dataset, vocab_dict)

    with open(testopt.out, 'w') as out:
        print('\n'.join(answers), file=out)

if __name__ == '__main__':
    main()


