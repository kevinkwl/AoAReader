import aoareader as reader
import torch
import time
import argparse

# torch.backends.cudnn.enabled=True

parser = argparse.ArgumentParser(description="train.py")

# train options

parser.add_argument('-data', default='data/preprocessed.pt',
                    help='Path to the *-train.pt file from preprocess.py, default value is \'data/preprocessed.pt\'')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_ACC.pt to 'models/' directory, where ACC is the
                    validation accuracy""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pre-trained model.""")

# model parameters

parser.add_argument('-gru_size', type=int, default=384,
                    help='Size of GRU hidden states')
parser.add_argument('-embed_size', type=int, default=384,
                    help='Word embedding sizes')


# optimization

parser.add_argument('-batch_size', type=int, default=32,
                    help='Maximum batch size')

parser.add_argument('-dropout', type=float, default=0.1,
                    help='Dropout probability; applied in bidirectional gru.')

parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')

parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')

parser.add_argument('-learning_rate', type=float, default=0.001,
                    help="""Starting learning rate. Adam is
                    used, this is the global learning rate.""")

parser.add_argument('-weight_decay', type=float, default=0.0001,
                    help="""weight decay (L2 penalty)""")

# GPU

parser.add_argument('-gpu', default=0, type=int,
                    help="which gpu to use. (0, 1...)")

# Log

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval (minibatches).")


opt = parser.parse_args()
print(opt)

if opt.gpu:
    torch.cuda.set_device(opt.gpu)



# data = torch.load(data_file)
#
# vocab_dict = data['dict']
# train_data = data['train']
# valid_data = data['valid']
#
# valid_dataset = reader.Dataset(valid_data, 32, False)
# train_dataset = reader.Dataset(train_data, 32, True)
# model = reader.AoAReader(vocab_dict, 0.1, 384, 384)
#
#
#
# if torch.cuda.is_available():
#     model.cuda()
#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#
# nParams = sum([p.nelement() for p in model.parameters()])
# print('* number of parameters: %d' % nParams)


def loss_func(answers, pred_answers, answer_probs):
    num_correct = (answers == pred_answers).sum().squeeze().data[0]
    loss = - torch.mean(torch.log(answer_probs))
    return loss.cuda(), num_correct
'''
def train():
    def trainEpoch():
        for i in range(len(train_dataset)):
            (docs, docs_len, doc_mask), (querys, querys_len, query_mask), answers, candidates = train_dataset[i]

            pred_answers, answer_prob = model(docs, docs_len, doc_mask, querys, querys_len, query_mask, answers=answers, candidates=candidates)

            loss, num_correct = loss_func(answers, pred_answers, answer_prob)
            loss.cuda()

            print(loss.data[0])

            optimizer.zero_grad()
            loss.backward()

            for parameter in model.parameters():
                parameter.grad.data.clamp_(-5.0, 5.0)
            optimizer.step()

    for i in range(2):
        trainEpoch()
train()

import time

def loss_func(answers, pred_answers, answer_probs):
    #num_correct = (answers == pred_answers).sum().squeeze().data[0]
    num_correct = 0
    loss = - torch.mean(torch.log(answer_probs))
    return loss.cuda(), num_correct
'''
def eval(model, data):
    total_loss = 0
    total = 0
    total_correct = 0

    model.eval()
    for i in range(len(data)):
        (batch_docs, batch_docs_len, doc_mask), (batch_querys, batch_querys_len, query_mask), batch_answers, candidates = data[i]

        pred_answers, probs = model(batch_docs, batch_docs_len, doc_mask,
                                    batch_querys, batch_querys_len, query_mask,
                                    answers=batch_answers, candidates=candidates)

        loss, num_correct = loss_func(batch_answers, pred_answers, probs)

        total_in_minibatch = batch_answers.size(0)
        total_loss += loss.data[0] * total_in_minibatch
        total_correct += num_correct
        total += total_in_minibatch

        del loss, pred_answers, probs

    model.train()
    return total_loss / total, total_correct / total


def trainModel(model, trainData, validData, optimizer: torch.optim.Adam):
    print(model)
    start_time = time.time()

    def trainEpoch(epoch):

        total_loss, total, total_num_correct = 0, 0, 0
        report_loss, report_total, report_num_correct = 0, 0, 0
        for i in range(len(trainData)):
            (batch_docs, batch_docs_len, doc_mask), (batch_querys, batch_querys_len, query_mask), batch_answers, candidates = trainData[i]

            model.zero_grad()
            pred_answers, answer_probs = model(batch_docs, batch_docs_len, doc_mask, batch_querys, batch_querys_len, query_mask,answers=batch_answers, candidates=candidates)

            loss, num_correct = loss_func(batch_answers, pred_answers, answer_probs)

            loss.backward()
            for parameter in model.parameters():
                parameter.grad.data.clamp_(-5.0, 5.0)
            # update the parameters
            optimizer.step()

            total_in_minibatch = batch_answers.size(0)

            report_loss += loss.data[0] * total_in_minibatch
            report_num_correct += num_correct
            report_total += total_in_minibatch

            total_loss += loss.data[0] * total_in_minibatch
            total_num_correct += num_correct
            total += total_in_minibatch
            if i % 50 == 0:
                print("Epoch %2d, %5d/%5d; avg loss: %.2f; acc: %6.2f;  %6.0f s elapsed" %
                      (epoch, i+1, len(trainData),
                       report_loss / report_total,
                       report_num_correct / report_total * 100,
                       time.time()-start_time))

                report_loss = report_total = report_num_correct = 0
            del loss, pred_answers, answer_probs

        return total_loss / total, total_num_correct / total

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch)
        print('Epoch %d:\t average loss: %.2f\t train accuracy: %g' % (epoch, train_loss, train_acc*100))

        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(model, validData)
        print('=' * 20)
        print('Evaluating on validation set:')
        print('Validation loss: %.2f' % valid_loss)
        print('Validation accuracy: %g' % (valid_acc*100))
        print('=' * 20)

        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'epoch': epoch,
            'optimizer': optimizer_state_dict,
            'opt': opt,
        }
        torch.save(checkpoint,
                   'models/%s_epoch%d_acc_%.2f.pt' % (opt.save_model, epoch, 100*valid_acc))

def main():

    print("Loading data from ", opt.data)
    data = torch.load(opt.data)

    vocab_dict = data['dict']
    train_data = data['train']
    valid_data = data['valid']

    train_dataset = reader.Dataset(train_data, opt.batch_size, opt.gpu)
    valid_dataset = reader.Dataset(valid_data, opt.batch_size, opt.gpu, volatile=True)

    print(' * vocabulary size = %d' %
          (vocab_dict.size()))
    print(' * number of training samples. %d' %
          len(data['train']['answers']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    model = reader.AoAReader(vocab_dict, dropout_rate=opt.dropout, embed_dim=opt.embed_size, hidden_dim=opt.gru_size)
    # no way on CPU
    model.cuda()

    if opt.train_from:
        checkpoint = torch.load(opt.train_from)
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        model.load_state_dict(chk_model)
        opt.start_epoch = checkpoint['epoch'] + 1

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    if opt.train_from:
        optimizer.load_state_dict(checkpoint['optimizer'])

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)


    trainModel(model, train_dataset, valid_dataset, optimizer)

if __name__ == '__main__':
    main()