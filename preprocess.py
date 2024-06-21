import os
from dictionaries import diacritics_list, symbol, classes, arabic_characters
import numpy as np
import random

def read_data(path_data):

    train = None
    with open(path_data + '/train.txt', 'r') as file:
        train = file.readlines()
    print('Training Samples', len(train))

    val = None
    with open(path_data + '/valid.txt', 'r') as file:
        valid = file.readlines()
    print('Validation Samples', len(valid))

    return train, valid

def remove_diac(raw):
    return raw.translate(str.maketrans('', '', ''.join(diacritics_list)))


def one_hot(samples, length):
    res_one_hot = []
    for elem in samples:
        ans = [0] * length
        ans[elem] = 1
        res_one_hot.append(ans)
    return res_one_hot

def spliter(text):
    text = text.replace('.', '.\n')
    text = text.replace(',', ',\n')
    text = text.replace('،', '،\n')
    text = text.replace(':', ':\n')
    text = text.replace(';', ';\n')
    text = text.replace('؛', '؛\n')
    text = text.replace('(', '\n(')
    text = text.replace(')', ')\n')
    text = text.replace('[', '\n[')
    text = text.replace(']', ']\n')
    text = text.replace('{', '\n{')
    text = text.replace('}', '}\n')
    text = text.replace('«', '\n«')
    text = text.replace('»', '»\n')
    return text.split('\n')


def split_data(data_raw):

    res = []
    
    for row in data_raw:

        line = spliter(row)

        for sub in line:

            count_character = len(remove_diac(sub).strip())

            if count_character > 0 and count_character <= 500:
                res.append(sub.strip())

    return res


def map_proc(new_data):

    sample, labels = [], []

    for line in new_data:
        x = [symbol['<SOS>']]
        y = [classes['<SOS>']]

        for id, char in enumerate(line):
            if char in diacritics_list:
                continue

            x.append(symbol[char])

            if char not in arabic_characters:
                y.append(classes[''])
            else:
                diac = ''
                if id + 1 < len(line) and line[id + 1] in diacritics_list:
                    diac = line[id + 1]
                    if id + 2 < len(line) and line[id + 2] in diacritics_list and diac + line[id + 2] in classes:
                        diac += line[id + 2]
                    elif id + 2 < len(line) and line[id + 2] in diacritics_list and line[id + 2] + diac in classes:
                        diac = line[id + 2] + diac
                y.append(classes[diac])

        assert(len(x) == len(y))

        x.append(symbol['<EOS>'])
        y.append(classes['<EOS>'])

        y = one_hot(y, len(classes))
        sample.append(x)
        labels.append(y)

    return sample, labels
