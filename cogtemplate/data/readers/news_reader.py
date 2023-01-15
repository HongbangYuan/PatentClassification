import os
from cogtemplate.data.readers.base_reader import BaseReader
from cogtemplate.data.datable import DataTable
from cogtemplate.utils.vocab_utils import Vocabulary
from tqdm import tqdm
from collections import Counter
# from cogtemplate.utils.download_utils import Downloader


class NewsReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.train_file = 'traindata.txt'
        self.dev_file = 'devdata.txt'
        self.test_file = 'testdata.txt'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.label_vocab = Vocabulary()


    def _read(self, path=None,isTraining=True):
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        for line in lines:
            label,sentence = line.split("\t")
            datable("label",label)
            datable("sentence",sentence)
            self.label_vocab.add(label)
        return datable

    def _read_train(self, path=None):
        return self._read(path,isTraining=True)

    def _read_dev(self, path=None):
        return self._read(path,isTraining=False)

    def _read_test(self, path=None):
        return self._read(path,isTraining=False)

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}


if __name__ == "__main__":
    reader = NewsReader(raw_data_path="/data/hongbang/projects/PatentClassification/datapath/text_classification/news/raw_data")

    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

