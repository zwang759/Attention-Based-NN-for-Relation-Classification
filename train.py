# codeing: utf-8

import argparse
import logging
import mxnet as mx
import numpy as np
import random
from mxnet import gluon, nd
from mxnet import autograd
from mxnet.gluon.data import DataLoader
import gluonnlp as nlp
from sklearn.metrics import f1_score, precision_score, recall_score
from load_data import load_dataset
from model import RelationClassifier
from utils import logging_config

relation_types = [
    "Component-Whole",
    "Component-Whole-Inv",
    "Instrument-Agency",
    "Instrument-Agency-Inv",
    "Member-Collection",
    "Member-Collection-Inv",
    "Cause-Effect",
    "Cause-Effect-Inv",
    "Entity-Destination",
    "Entity-Destination-Inv",
    "Content-Container",
    "Content-Container-Inv",
    "Message-Topic",
    "Message-Topic-Inv",
    "Product-Producer",
    "Product-Producer-Inv",
    "Entity-Origin",
    "Entity-Origin-Inv",
    "Other"
    ]

parser = argparse.ArgumentParser(
    description='Train a (short) text classifier - via convolutional or other standard architecture')
parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data',
                    default='cleaned_entity_type_train.tsv')
#ChunkTrain
#cleanTrain.tsv
#cleaned_entity_type_train.tsv
#crawl-300d-2M
parser.add_argument('--epochs', type=int, default=25, help='Upper epoch limit')
parser.add_argument('--optimizer', type=str, help='Optimizer (adam, sgd, etc.)', default='nadam')
parser.add_argument('--lr', type=float, help='Learning rate', default=0.0001)
parser.add_argument('--batch_size', type=int, help='Training batch size', default=10)
parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.5)
parser.add_argument('--embedding_source', type=str, default='crawl-300d-2M', help='Pre-trained embedding source name')
parser.add_argument('--log_dir', type=str, default='.', help='Output directory for log file')
parser.add_argument('--fixed_embedding', action='store_true', help='Fix the embedding layer weights')
parser.add_argument('--random_embedding', action='store_true', help='Use random initialized embedding layer')

np.random.seed(10)
random.seed(10)
mx.random.seed(10)
args = parser.parse_args()
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()


def train_classifier(model, transformer, data_train, data_test, cross_validation=True, test_transformer=None):

    data_train = gluon.data.SimpleDataset(data_train).transform(transformer)
    if cross_validation:
        data_test = gluon.data.SimpleDataset(data_test).transform(transformer)
    else:
        data_test = gluon.data.SimpleDataset(data_test).transform(test_transformer)

    train_dataloader = DataLoader(dataset=data_train, batch_size=args.batch_size)
    test_dataloader = DataLoader(dataset=data_test, batch_size=args.batch_size)

    # model.hybridize() ## OPTIONAL for efficiency - perhaps easier to comment this out during debugging

    trainer = gluon.Trainer(model.collect_params(), 'nadam', {'learning_rate': args.lr})

    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, x in enumerate(train_dataloader):
            data, label, inds = x
            with autograd.record():
                output = model(data, inds)
                l = loss_fn(output, label).mean()
            l.backward()
            trainer.step(data.shape[0])  ## step based on batch size
            epoch_loss += l.asscalar()
        logging.info("Epoch {}'s loss = {}".format(epoch + 1, epoch_loss))
        if cross_validation:
            if epoch > 3:
                test_accuary, test_loss, y_pred, y_true = evaluate(model, test_dataloader)
                logging.info("Epoch {}'s Test acc = {}, Test Loss = {}".format(epoch + 1,test_accuary, test_loss))
                logging.info("Micro Precision Score: {}".format(precision_score(y_true, y_pred, average='macro')))
                logging.info("Micro Recall Score: {}".format(recall_score(y_true, y_pred, average='macro')))
                logging.info("Micro F1 Score: {}".format(f1_score(y_true, y_pred, average='macro')))
                logging.info("Precision Score for each class: {}".format(precision_score(y_true, y_pred, average=None)))
                logging.info("Recall Score for each class: {}".format(recall_score(y_true, y_pred, average=None)))
                logging.info("F1 Score for each class: {}".format(f1_score(y_true, y_pred, average=None)))
            if epoch_loss < 350:
                print('check point')
                return test_accuary
        else:
            if epoch_loss < 350:
                predict_test_label(model, test_dataloader, ctx=mx.cpu())


def predict_test_label(model, test_dataloader, ctx=mx.cpu()):
    """
    Get predictions on the dataloader items from model
    Write predicted label into a txt file
    """
    with open('test_predicted.txt', 'w', newline='') as f_output:
        for i, (data, inds) in enumerate(test_dataloader):
            out = model(data, inds)
            for j in range(out.shape[0]):
                probs = mx.nd.softmax(out[j]).asnumpy()
                best_probs = np.argmax(probs)
                predicted_label = relation_types[int(best_probs)]
                f_output.write(predicted_label)
                f_output.write('\n')


def evaluate(model, dataloader, ctx=mx.cpu()):
    """
    Get predictions on the dataloader items from model
    Return metrics (accuracy, etc.)
    """
    acc = 0.
    avg_loss = 0.
    total_loss = 0.
    total_sample_num = 0.
    total_correct_num = 0.
    y_pred = []
    y_true = []
    for i, (data, label, inds) in enumerate(dataloader):
        out = model(data, inds)
        l = loss_fn(out, label).mean()
        total_loss += l.asscalar()

        for j in range(out.shape[0]):
            probs = mx.nd.softmax(out[j]).asnumpy()
            lab = int(label[j].asscalar())
            best_probs = np.argmax(probs)
            y_pred.append(best_probs)
            y_true.append(lab)
            if lab == best_probs:
                total_correct_num += 1.
            total_sample_num += 1.

    acc = total_correct_num / total_sample_num

    return acc, total_loss, y_pred, y_true


def k_fold_cross_valid(k, transformer, all_dataset, vocab, max_length):
    test_acc = []
    # divided by k folds
    fold_size = len(all_dataset) // k
    random.shuffle(all_dataset)
    for test_i in range(k):
        model = model_initializer(vocab, max_length)
        # print the network
        print(model)
        test_data = all_dataset[test_i * fold_size: (test_i + 1) * fold_size]
        train_data = all_dataset[: test_i * fold_size] + all_dataset[(test_i + 1) * fold_size:]
        print(len(train_data), len(test_data))
        test_acc.append(train_classifier(model, transformer, train_data, test_data))
        logging.info("Finished {} iteration of {}-Fold Cross Validation".format(test_i + 1, k))
    print('average acc for test data: ', sum(test_acc) / k)


def model_initializer(vocab, max_length):
    """
    input: vocab, max_length of sentence
    return: model with intialization
    """
    if args.embedding_source:
        pretrained_embedding = nlp.embedding.create('fasttext', source=args.embedding_source)
        vocab.set_embedding(pretrained_embedding)
        for word in vocab.embedding.idx_to_token:
            if (vocab.embedding[word] == nd.zeros(300)).sum() == 300:
                vocab.embedding[word] = nd.random.normal(0, 1, 300)
    emb_input_dim, emb_output_dim = vocab.embedding.idx_to_vec.shape
    model = RelationClassifier(emb_input_dim, emb_output_dim, dropout=args.dropout, max_seq_len=max_length)
    model.initialize(mx.init.Xavier(), ctx=ctx)
    if not args.random_embedding:
        model.embedding.weight.set_data(
            vocab.embedding.idx_to_vec)  ## set the embedding layer parameters to pre-trained embedding
    elif args.fixed_embedding:
        model.embedding.collect_params().setattr('grad_req', 'null')

    return model


if __name__ == '__main__':

    logging_config(args.log_dir, 'train', level=logging.INFO)
    ctx = mx.cpu()  ## or mx.gpu(N) if GPU device N is available


    # for cross-validation
    vocab, train_dataset, transformer, max_length = load_dataset(args.train_file)
    print('sentence length is: ', max_length)
    # k-fold cross validation for training and testing different models
    k_fold_cross_valid(5, transformer, train_dataset, vocab, max_length)


    # for predicting the test labels, uncomment the codes below and comment the above lines for cross-validation
    # vocab, train_dataset, test_dataset, train_transformer, test_transformer, max_length = load_dataset(args.train_file,
    #                                                                                                    cross_validation=False,
    #                                                                                                    test_file='cleaned_entity_type_test.tsv')
    # random.shuffle(train_dataset)
    # print('sentence length is: ', max_length)
    # model = model_initializer(vocab, max_length)
    # print(model)
    #
    # train_classifier(model, train_transformer, train_dataset, test_dataset, cross_validation=False, test_transformer=test_transformer)