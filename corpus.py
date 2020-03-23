import pickle


class Corpus:
    def __init__(self, path, name=None):
        with open(path, 'rb') as f:
            self._data = pickle.load(f)
        self._path = path
        self._name = name
        self._ix = 0

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value
        return self

    def save(self, path=None):
        path = self._path if not path else path
        with open(path, 'wb') as f:
            pickle.dump(self._data, f)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return self

    def __next__(self):
        if self._ix == self.__len__():
            self._ix = 0
            raise StopIteration()
        ret = self._ix
        self._ix += 1
        return self._data[ret]


if __name__ == '__main__':
    corpus = Corpus('ner_data/corpus_test.pkl')
    for i, sample in enumerate(corpus):
        print(i, sample)
    for i, sample in enumerate(corpus):
        sample['test_key'] = 'test'
    print(corpus[0]['test_key'])
