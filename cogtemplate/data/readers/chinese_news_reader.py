import os
from cogtemplate.data.readers.base_reader import BaseReader
from cogtemplate.data.datable import DataTable
from cogtemplate.utils.vocab_utils import Vocabulary
from tqdm import tqdm
from collections import Counter
# from cogtemplate.utils.download_utils import Downloader


class ChieseNewsReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        # downloader = Downloader()
        # downloader.download_sst2_raw_data(raw_data_path)
        self.train_file = 'train.txt'
        self.dev_file = 'dev.txt'
        self.test_file = 'test.txt'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.label_vocab = Vocabulary()
        self.word_vocab = Vocabulary()
        self.ext_vocab = ["<pad>", "<unk>", "<go>", "<eos>"]
        self.vocab_list = []
        self.vocab_counter = Counter()

    def _read(self, path=None,isTraining=True):
        print("Reading data...")
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        cut = 10 * 10000 if isTraining else 10000
        for line in tqdm(lines[:cut]):
            words = list(line.strip())
            datable("words",["<go>"]+words)
            self.vocab_counter += Counter(words)
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
        self.vocab_counter = self.vocab_counter.most_common()
        offset = len(self.ext_vocab)
        ext_vocab_dict = {value:key for key,value in enumerate(self.ext_vocab)}

        vocab_dict = {value[0]:key+offset for key,value in enumerate(self.vocab_counter)}
        predifined_vocab_dict = {**ext_vocab_dict,**vocab_dict}
        self.word_vocab.add_dict(predifined_vocab_dict)
        self.word_vocab.create()
        return {"word_vocab": self.word_vocab}


if __name__ == "__main__":
    reader = ChieseNewsReader(raw_data_path="/data/hongbang/projects/PatentClassification/datapath/language_models/chinese_news/raw_data")
    cache_file = "/data/hongbang/projects/PatentClassification/datapath/language_models/chinese_news/cache/reader_datas.pkl"

    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    from cogtemplate.utils.io_utils import load_pickle,save_pickle

    save_pickle([train_data, dev_data, test_data, vocab], cache_file)
    print("end")
