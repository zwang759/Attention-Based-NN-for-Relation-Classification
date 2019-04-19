# known issues, when you replace the identified entity by its entity type
# if this identified entity is either the first token or the last token of the sentence
# then you may need to look at the output data to see if you need to add double quote mark(")before or after it.
# updated, add line206 to line209 to handle this problem

from flair.models import SequenceTagger
from flair.data import Sentence
import csv
import io

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


def clean_data(tr_array, tagger):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    with open('EntityTypeTrain.tsv', 'w', newline='') as f_output:
        for i, instance in enumerate(tr_array):
            label, e1, e2, text = instance

            tokens = text.split(' ')

            junk_indices = []

            # get a Flair sentence object by default whitespace tokenization
            Sentence_object = Sentence(text)
            # get predicted NER tags
            tagger.predict(Sentence_object)

            new_e1 = e1
            new_e2 = e2

            # if this sentence is detected to have NER tags
            if len(Sentence_object.get_spans('ner')) > 0:
                # entity is a span of entities in the form of ' "NER tags" [index1, index2, ...(index start from 1)] "token1, ..." '
                # data needs to be cleaned
                for entity in Sentence_object.get_spans('ner'):
                    raw_string_list = str(entity).split(' ')
                    entity_type, messy_index = raw_string_list[0], raw_string_list[1]
                    # ignore MISC category
                    if entity_type != 'MISC-span':
                        # get clean index
                        clean_index = messy_index[1:len(messy_index) - 2]
                        clean_index = clean_index.split(',')
                        # remember to -1 when indexing
                        # k is the index of the first token of marked entity type in a token span
                        k = int(clean_index[0]) - 1
                        # replace the first marked token by its entity type
                        tokens[k] = entity_type
                        # if any token span is detected
                        if len(clean_index) > 1:
                            # from the 2nd marked token to the n-th marked token
                            for p in clean_index[1: len(clean_index)]:
                                if p not in junk_indices:
                                    # add their indices to junk list
                                    # remember to -1 when indexing
                                    junk_indices.append(int(p)-1)
                if junk_indices:
                    for index in sorted(junk_indices, reverse=True):
                        # be careful about e1, e2's new value after deleting junk tokens
                        if index < e1:
                            new_e1 = new_e1 - 1
                        if index < e2:
                            new_e2 = new_e2 - 1
                        del tokens[index]
            print((label, new_e1, new_e2, ' '.join(tokens)))

            tsv_output = csv.writer(f_output, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='', quotechar='')
            tsv_output.writerow((label, new_e1, new_e2, ' '.join(tokens)))


def add_Chunktag(tr_array, tagger):
    """
    Inputs: array and flair tagger
    Write into ChunkTrain.tsv file
    """
    punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    with open('ChunkTrain.tsv', 'w', newline='') as f_output:
        for i, instance in enumerate(tr_array):
            label, e1, e2, text = instance
            tokens = text.split(' ')
            key_token1 = tokens[e1]
            key_token2 = tokens[e1]
            if tokens[e1][-1] != "'" or tokens[e1][-2] != 's':
                key_token1 = tokens[e1].split("'")[0].translate(str.maketrans('', '', punctuation))
            if tokens[e2][-1] != "'" or tokens[e2][-2] != 's':
                key_token2 = tokens[e2].split("'")[0].translate(str.maketrans('', '', punctuation))                      
            Sentence_object = Sentence(text, use_tokenizer=True)
            tagger.predict(Sentence_object)
            new_text = Sentence_object.to_tagged_string()
            new_tokens = new_text.split(' ')
            offset1 = new_tokens[e1:].index(key_token1)
            offset2 = new_tokens[e2:].index(key_token2)
            new_e1 = e1 + offset1
            new_e2 = e2 + offset2

            tsv_output = csv.writer(f_output, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='', quotechar='')
            tsv_output.writerow((label, new_e1, new_e2, ' '.join(new_tokens)))


def add_Chunktag_and_tokenize(text, e1, e2, tagger):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    tokens = text.split(' ')
    key_token1 = tokens[e1]
    key_token2 = tokens[e1]
    if tokens[e1][-1] != "'" or tokens[e1][-2] != 's':
        key_token1 = tokens[e1].split("'")[0].translate(str.maketrans('', '', punctuation))
    if tokens[e2][-1] != "'" or tokens[e2][-2] != 's':
        key_token2 = tokens[e2].split("'")[0].translate(str.maketrans('', '', punctuation))
    Sentence_object = Sentence(text, use_tokenizer=True)
    tagger.predict(Sentence_object)
    new_text = Sentence_object.to_tagged_string()
    new_tokens = new_text.split(' ')
    offset1 = new_tokens[e1:].index(key_token1)
    offset2 = new_tokens[e2:].index(key_token2)
    new_e1 = e1 + offset1
    new_e2 = e2 + offset2
    return new_tokens, new_e1, new_e2


def clean_test_data(test_array, tagger):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    with open('EntityTypeTest.tsv', 'w', newline='') as f_output:
        for i, instance in enumerate(test_array):
            e1, e2, text = instance

            tokens = text.split(' ')

            junk_indices = []

            # get a Flair sentence object by default whitespace tokenization
            Sentence_object = Sentence(text)
            # get predicted NER tags
            tagger.predict(Sentence_object)

            new_e1 = e1
            new_e2 = e2

            # if this sentence is detected to have NER tags
            if len(Sentence_object.get_spans('ner')) > 0:
                # entity is a span of entities in the form of ' "NER tags" [index1, index2, ...(index start from 1)] "token1, ..." '
                # data needs to be cleaned
                for entity in Sentence_object.get_spans('ner'):
                    raw_string_list = str(entity).split(' ')
                    entity_type, messy_index = raw_string_list[0], raw_string_list[1]
                    # ignore MISC category
                    if entity_type != 'MISC-span':
                        # get clean index
                        clean_index = messy_index[1:len(messy_index) - 2]
                        clean_index = clean_index.split(',')
                        # remember to -1 when indexing
                        # k is the index of the first token of marked entity type in a token span
                        k = int(clean_index[0]) - 1
                        # replace the first marked token by its entity type
                        tokens[k] = entity_type
                        # if any token span is detected
                        if len(clean_index) > 1:
                            # from the 2nd marked token to the n-th marked token
                            for p in clean_index[1: len(clean_index)]:
                                if p not in junk_indices:
                                    # add their indices to junk list
                                    # remember to -1 when indexing
                                    junk_indices.append(int(p)-1)
                if junk_indices:
                    for index in sorted(junk_indices, reverse=True):
                        # be careful about e1, e2's new value after deleting junk tokens
                        if index < e1:
                            new_e1 = new_e1 - 1
                        if index < e2:
                            new_e2 = new_e2 - 1
                        del tokens[index]
            entity_type = ['LOC-span', 'ORG-span', 'PER-span']
            if tokens[0] == '"' and tokens[-1] in entity_type:
                tokens[-1] = tokens[-1] + '."'
            if tokens[-1] == '"' and tokens[0] in entity_type:
                tokens[0] = tokens[0] + '"'
            if tokens[0] != '"' and tokens[-1] in entity_type:
                tokens[-1] = tokens[-1] + '.'
            print((new_e1, new_e2, ' '.join(tokens)))

            tsv_output = csv.writer(f_output, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='', quotechar='')
            tsv_output.writerow((new_e1, new_e2, ' '.join(tokens)))

# add NER to train
train_array = load_tsv_to_array('cleanTrain.tsv')
tagger = SequenceTagger.load('ner')
clean_data(train_array, tagger)

#add_POStag to train
# train_array = load_tsv_to_array('cleanTrain.tsv')
# tagger = SequenceTagger.load('chunk')
# add_Chunktag(train_array,tagger)

# add NER to test
test_array = load_tsv_to_array('cleanTest.tsv',IsTest=True)
tagger = SequenceTagger.load('ner')
clean_test_data(test_array, tagger)