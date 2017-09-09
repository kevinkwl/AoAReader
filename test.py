import aoareader as reader
import torch
import time
import argparse
import os

from preprocess import get_stories, vectorize_stories


parser = argparse.ArgumentParser(description="test.py")

parser.add_argument('-testdata', default='data/test.txt.pt',
                    help='Path to the test.txt.pt, test.txt.pt will be used if exists.')

parser.add_argument('-dict', default="data/dict.pt",
                    help='Path to the dictionary file, default value: data/dict.pt')

parser.add_argument('-out', default='data/result.txt',
                    help='output file name.')

parser.add_argument('-model', required=True, help='path to the saved model.')


testopt = parser.parse_args()
print(testopt)


def load_testdata(testfile, vocab_dict, with_answer=True):
    if os.path.exists(testfile + '.pt'):
        return torch.load(testfile + '.pt')
    else:
        testd = {}
        with open(testfile, 'r') as tf:
            tlines = tf.readlines()
            test_stories = get_stories(tlines, with_answer=with_answer)
            testd['documents'], testd['querys'], testd['answers'], testd['candidates'] = vectorize_stories(test_stories, vocab_dict)
        torch.save(testd, testfile + '.pt')
        return testd

def evalulate(model, data, vocab_dict):

    def acc(answers, pred_answers):
        num_correct = (answers == pred_answers).sum().squeeze().data[0]
        return num_correct

    model.eval()
    answers = []
    total_correct = 0
    total = 0
    for i in range(len(data)):
        (batch_docs, batch_docs_len, doc_mask), (batch_querys, batch_querys_len, query_mask), batch_answers , candidates = data[i]

        pred_answers, _ = model(batch_docs, batch_docs_len, doc_mask,
                                    batch_querys, batch_querys_len, query_mask,
                                    candidates=candidates, answers=batch_answers)

        answers.extend(pred_answers.data)
        num_correct = acc(batch_answers, pred_answers)

        total_in_minibatch = batch_answers.size(0)
        total_correct += num_correct
        total += total_in_minibatch
        del pred_answers

    print("Evaluating on test set:\nAccurary {:.2%}".format(total_correct / total))
    return vocab_dict.convert2word(answers)

def main():
    print("Loading dict", testopt.dict)
    vocab_dict = torch.load(testopt.dict)

    print("Loading test data")
    test_data = torch.load(testopt.testdata)

    print("Loading model from ", testopt.model)
    ckp = torch.load(testopt.model)

    opt = ckp['opt']
    model_state = ckp['model']

    if opt.gpu:
        torch.cuda.set_device(opt.gpu)

    test_dataset = reader.Dataset(test_data, opt.batch_size, True, volatile=True)

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


