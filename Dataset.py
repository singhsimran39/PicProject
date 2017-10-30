import numpy as np

class Dataset:

    def __init__(self, data, labels):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._labels = labels
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0 and shuffle:
            idx = np.arange(self._num_examples)
            np.random.shuffle(idx)
            self._data = self.data[idx]
            self._labels = self.labels[idx]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            remaining_examples = self._num_examples - start
            remaining_data = self.data[start:self._num_examples]
            remaning_labels = self.labels[start:self._num_examples]
            start = 0
            self._index_in_epoch = 0
            return remaining_data, remaning_labels
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end]
