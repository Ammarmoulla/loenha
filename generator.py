import numpy as np
from keras.utils import Sequence
from preprocess import map_proc, one_hot
from dictionaries import symbol, classes

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