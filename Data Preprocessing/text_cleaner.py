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


def clean_text(text):
    if text[0] == '"' and text[1] == ' ':
        text = text[0:1] + text[2:len(text)]
    return text


# clean train data file and write into a new file
# uncomment the following codes if you want to clean the train data file
# array = load_tsv_to_array('semevalTrain_fixed.tsv')
# with open('cleanTrain.tsv', 'w', newline='') as f_output:
#     for i, instance in enumerate(array):
#         label, e1, e2, text = instance
#         newText = clean_text(text)
#         tsv_output = csv.writer(f_output, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='', quotechar='')
#         tsv_output.writerow((label, e1, e2, newText))

# clean test data file and write into a new file
# uncomment the following codes if you want to clean the test data file
# array = load_tsv_to_array('semevalTest_fixed_nolabels.tsv', IsTest=True)
# with open('cleanTest.tsv', 'w', newline='') as f_output:
#     for i, instance in enumerate(array):
#         e1, e2, text = instance
#         newText = clean_text(text)
#         tsv_output = csv.writer(f_output, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='', quotechar='')
#         tsv_output.writerow((e1, e2, newText))
