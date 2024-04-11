import numpy as np
from keras.utils import Sequence
from preprocess import map_proc, one_hot, split_data, remove_diac
from dictionaries import symbol, classes
import random

class Generator(Sequence):
    def __init__(self, rows, bz):
        self.rows = rows
        self.batch_size = bz

    def __len__(self):
        return int(np.ceil(len(self.rows) / float(self.batch_size)))

    def __getitem__(self, idx):
        rows = self.rows[idx * self.batch_size:(idx + 1) * self.batch_size]
        samples, labels = map_proc(rows)

        max_len_sample = np.max([len(x) for x in samples])
        max_len_labels = np.max([len(y) for y in labels])

        assert(max_len_sample == max_len_labels)

        X = list()
        for x in samples:
            tmp = list(x)
            tmp.extend([symbol['<PAD>']] * (max_len_sample - len(tmp)))
            X.append(np.asarray(tmp))

        Y = list()
        for y in labels:
            tmp = list(y)
            tmp.extend(one_hot([classes['<PAD>']] * (max_len_labels - len(tmp)), len(classes)))
            Y.append(np.asarray(tmp))

        X_batch = np.asarray(X)
        Y_batch = np.asarray(Y)

        return X_batch, Y_batch
    

def full_process(train, valid, batch_size, shuffle):
    
    train_data = split_data(train)
    valid_data = split_data(valid)

    print('the number samples of train: {}'.format(len(train_data)))
    print('the number samples of valid: {}'.format(len(valid_data)))

    if shuffle:
        random.shuffle(train_data)
        random.shuffle(valid_data)
        
    train_split = list(sorted(train_data, key=lambda line: len(remove_diac(line))))
    val_split = list(sorted(valid_data, key=lambda line: len(remove_diac(line))))

    train_generator = Generator(train_split, batch_size)
    valid_generator = Generator(val_split, batch_size)

    return train_generator, valid_generator