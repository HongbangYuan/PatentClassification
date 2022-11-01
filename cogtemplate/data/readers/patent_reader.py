import os
from cogtemplate.data.readers.base_reader import BaseReader
from cogtemplate.data.datable import DataTable
from cogtemplate.utils.vocab_utils import Vocabulary
import json
from random import shuffle


# from cogtemplate.utils.download_utils import Downloader


class PatentReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        # self.train_file = 'train.json'
        self.train_file = 'augmented_train.json'
        # self.dev_file = 'dev.tsv'
        self.test_file = 'testA.json'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        # self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.label_vocab = Vocabulary()

    def _read_train_and_dev(self,path=None,split=0.9):
        print("Reading data...")
        datable = DataTable()
        with open(path,'r',encoding='utf8') as file:
            lines = file.readlines()
        shuffle(lines)
        my_split = 0.9 if split is None else split
        divide = int(my_split * len(lines))
        train_lines = lines[:divide]
        dev_lines = lines[divide:]
        train_datable = DataTable()
        dev_datable = DataTable()

        if split is None:
            train_lines = lines
        for line in train_lines:
            dict_data = json.loads(line)
            for key in ["id","title","assignee","abstract"]:
                train_datable(key,dict_data[key])
            train_datable("label",dict_data["label_id"])
            self.label_vocab.add(dict_data["label_id"])

        for line in dev_lines:
            dict_data = json.loads(line)
            for key in ["id","title","assignee","abstract"]:
                dev_datable(key,dict_data[key])
            dev_datable("label",dict_data["label_id"])
            self.label_vocab.add(dict_data["label_id"])

        return train_datable,dev_datable

    # def _read_train(self, path=None):
    #     return self._read(path)
    #
    # def _read_dev(self, path=None):
    #     return self._read(path)
    #
    def _read_test(self, path=None):
        print("Reading data...")
        datable = DataTable()
        with open(path,'r',encoding='utf8') as file:
            lines = file.readlines()

        for line in lines:
            dict_data = json.loads(line)
            for key in ["id","title","assignee","abstract"]:
                datable(key,dict_data[key])
            # datable("label",dict_data["label_id"])
            # self.label_vocab.add(dict_data["label_id"])
        return datable

    def read_all(self,split=0.9):
        train_data,dev_data = self._read_train_and_dev(self.train_path,split=split)
        test_data = self._read_test(self.test_path)
        return train_data,dev_data,test_data
        # return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}


if __name__ == "__main__":
    reader = PatentReader(raw_data_path="/data/hongbang/CogAGENT/datapath/text_classification/patent/raw_data")
    # train_data = reader._read_train()
    train_data, dev_data,test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
