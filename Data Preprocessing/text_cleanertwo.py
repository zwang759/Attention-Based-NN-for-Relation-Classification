# This file is a fix for the issues faced in data_preprocessing.py
# This might be useless after the update on data_preprocessing.py
# Dont use this

import csv
import io
import re


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


def clean_text(text):
    new_text = text
    if text[0] != '"' and text[-1] == '"':
        new_text = '"' + text[0:len(text)]
    if text[0] == '"' and text[-1] != '"':
        new_text = text[0:len(text)] + '"'
    if new_text[-1] != '"' and new_text[-1] != '.':
        new_text = new_text + '.'
    return new_text

# for train
# array = load_tsv_to_array('EntityTypeTrain.tsv')
# with open('cleaned_entity_type_train.tsv', 'w', newline='') as f_output:
#     for i, instance in enumerate(array):
#         label, e1, e2, text = instance
#         newText = clean_text(text)
#         tsv_output = csv.writer(f_output, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='', quotechar='')
#         tsv_output.writerow((label, e1, e2, newText))

# for test
array = load_tsv_to_array('EntityTypeTest.tsv', IsTest=True)
with open('cleaned_entity_type_test.tsv', 'w', newline='') as f_output:
    for i, instance in enumerate(array):
        e1, e2, text = instance
        newText = clean_text(text)
        tsv_output = csv.writer(f_output, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='', quotechar='')
        tsv_output.writerow((e1, e2, newText))