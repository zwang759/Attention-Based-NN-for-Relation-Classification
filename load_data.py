# coding: utf-8

import gluonnlp as nlp
import mxnet as mx
import io
from segtok.segmenter import split_single
from segtok.tokenizer import split_contractions
from segtok.tokenizer import word_tokenizer


def load_tsv_to_array(fname, IsTest=False):
    """
    Inputs: file path
    Outputs: list/array of 3-tuples, each representing a data instance
    """
    if IsTest:
        arr = []
        with io.open(fname, 'r') as fp:
            for line in fp:
                els = line.split('\t')
                els[2] = els[2].strip()
                els[1] = int(els[1])
                els[0] = int(els[0])
                arr.append(tuple(els))
        return arr
    else:
        arr = []
        with io.open(fname, 'r') as fp:
            for line in fp:
                els = line.split('\t')
                els[3] = els[3].strip()
                els[2] = int(els[2])
                els[1] = int(els[1])
                arr.append(tuple(els))
        return arr


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


def tokenize(text):
    """
    Inputs: txt
    Outputs: tokens tokenized by segtok.tokenizer
    """
    tokens = []
    sentences = split_single(text)
    for sentence in sentences:
        contractions = split_contractions(word_tokenizer(sentence))
        tokens.extend(contractions)
    return tokens


def retokenize(text, e1, e2):
    """
    Inputs: txt, e1, e2
    Outputs: tokens, new e1, and new e2 after retokenization
    """
    punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    whitespace_tokens = text.split(' ')
    segtok_tokens = tokenize(text)
    key_token1 = whitespace_tokens[e1]
    key_token2 = whitespace_tokens[e2]
    if whitespace_tokens[e1][-1] != "'" or whitespace_tokens[e1][-2] != 's':
        if whitespace_tokens[e1].split("'")[0] != '':
            key_token1 = whitespace_tokens[e1].split("'")[0].translate(str.maketrans('', '', punctuation))
        else:
            key_token1 = whitespace_tokens[e1].split("'")[1].translate(str.maketrans('', '', punctuation))
    if whitespace_tokens[e2][-1] != "'" or whitespace_tokens[e2][-2] != 's':
        if whitespace_tokens[e2].split("'")[0] != '':
            key_token2 = whitespace_tokens[e2].split("'")[0].translate(str.maketrans('', '', punctuation))
        else:
            key_token2 = whitespace_tokens[e2].split("'")[1].translate(str.maketrans('', '', punctuation))
    offset1 = segtok_tokens[e1:].index(key_token1)
    offset2 = segtok_tokens[e2:].index(key_token2)
    new_e1 = e1 + offset1
    new_e2 = e2 + offset2
    return segtok_tokens, new_e1, new_e2


###    - Parse the input data by getting the word sequence and the argument POSITION IDs for e1 and e2
###    [[w_1, w_2, w_3, .....], [pos_1, pos_2], [label_id]]  for EACH data instance/sentence/argpair
def load_dataset(train_file, cross_validation=True, test_file=None):
    """
    Inputs: training, validation and test files in TSV format
    Outputs: vocabulary (with attached embedding), training, validation and test datasets ready for neural net training
    """
    train_array = load_tsv_to_array(train_file)
    if cross_validation == True:
        vocabulary, max_length, train_array = build_vocabulary(train_array)
        all_dataset = preprocess_dataset(train_array, vocabulary)
        data_transform = BasicTransform(relation_types, max_length)
        return vocabulary, all_dataset, data_transform, max_length
    else:
        test_array = load_tsv_to_array(test_file, IsTest=True)
        vocabulary, max_length, train_array, test_array = build_vocabulary(train_array, cross_validation, test_array)
        train_dataset = preprocess_dataset(train_array, vocabulary)
        test_dataset = preprocess_dataset(test_array, vocabulary, IsTest=True)
        train_data_transform = BasicTransform(relation_types, max_length)
        test_data_transform = TestDataTransform(max_length)
        return vocabulary, train_dataset, test_dataset, train_data_transform, test_data_transform, max_length


def build_vocabulary(tr_array, cross_validation=True, test_array=None):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    all_tokens = []
    max_len = 0
    # max_dist = 0
    for i, instance in enumerate(tr_array):
        label, e1, e2, text = instance
        tokens, new_e1, new_e2 = retokenize(text, e1, e2)
        # tokens, new_e1, new_e2 = text.split(' '), e1, e2
        if len(tokens) > max_len:
            max_len = len(tokens)
        # if e2 - e1 + 1 > max_dist:
        #     max_dist = e2 - e1 + 1
        tr_array[i] = (label, new_e1, new_e2, tokens)
        all_tokens.extend(tokens)
    if cross_validation:
        counter = nlp.data.count_tokens(all_tokens)
        vocab = nlp.Vocab(counter)
        return vocab, max_len, tr_array
    else:
        for i, instance in enumerate(test_array):
            e1, e2, text = instance
            tokens, new_e1, new_e2 = retokenize(text, e1, e2)
            # tokens, new_e1, new_e2 = text.split(' '), e1, e2
            if len(tokens) > max_len:
                max_len = len(tokens)
            # if e2 - e1 + 1 > max_dist:
            #     max_dist = e2 - e1 + 1
            test_array[i] = (new_e1, new_e2, tokens)
            all_tokens.extend(tokens)
        counter = nlp.data.count_tokens(all_tokens)
        vocab = nlp.Vocab(counter)
        return vocab, max_len, tr_array, test_array


def _preprocess(x, vocab, IsTest=False):
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    if IsTest:
        ind1, ind2, text_tokens = x
        data = vocab[text_tokens]
        return ind1, ind2, data
    else:
        label, ind1, ind2, text_tokens = x
        data = vocab[text_tokens]   ## map tokens (strings) to unique IDs
        # positional_encoding = np.arange(len(data))
        # positional_encoding[ind1:ind2+1] = 0
        # positional_encoding[:ind1] = positional_encoding[:ind1] - ind1
        # positional_encoding[ind2+1:] = positional_encoding[ind2+1:] - ind2
        # print(positional_encoding.tolist())
        return label, ind1, ind2, data


def preprocess_dataset(dataset, vocab, IsTest=False):
    preprocessed_dataset = [_preprocess(x, vocab, IsTest) for x in dataset]
    return preprocessed_dataset


class BasicTransform(object):
    """
    This is a callable object used by the transform method for a training dataset. It will be
    called during data loading/iteration.  

    Parameters
    ----------
    labels : list string
        List of the valid strings for classification labels
    max_len : int
        Maximum sequence length: shorter ones padded
    """
    def __init__(self, labels, max_len):
        # self._max_dist = max_dist
        self._max_seq_length = max_len
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i
    
    def __call__(self, label, ind1, ind2, data):
        label_id = self._label_map[label]
        padded_data = data + [0] * (self._max_seq_length - len(data))
        # padded_inds = np.arange(ind1, ind2).tolist() + [0] * (self._max_dist - (ind2 - ind1 + 1))
        # padded_inds = mx.nd.array(padded_inds)
        # padded_inds = mx.nd.arange(ind1,ind2)
        inds = mx.nd.array([ind1, ind2])
        # new_inds = mx.nd.array([0, ind2 - ind1])
        # valid_length = ind2 - ind1 + 1
        return mx.nd.array(padded_data, dtype='int32'), mx.nd.array([label_id], dtype='int32'), inds


class TestDataTransform(object):
    """
    This is a callable object used by the transform method for a test dataset. It will be
    called during data loading/iteration.

    Parameters
    ----------
    max_len : int,
        Maximum sequence length: shorter ones padded
    """

    def __init__(self, max_len):
        # self._max_dist = max_dist
        self._max_seq_length = max_len
        self._label_map = {}

    def __call__(self, ind1, ind2, data):
        padded_data = data + [0] * (self._max_seq_length - len(data))
        inds = mx.nd.array([ind1, ind2])
        return mx.nd.array(padded_data, dtype='int32'), inds