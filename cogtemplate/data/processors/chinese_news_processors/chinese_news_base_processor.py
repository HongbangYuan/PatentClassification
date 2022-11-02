from cogtemplate.data.datable import DataTable
from cogtemplate.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogtemplate.data.processors.base_processor import BaseProcessor
from torch import Tensor
import torch
transformers.logging.set_verbosity_error()  # set transformers logging level


class ChineseNewsForLMProcessor(BaseProcessor):
    def __init__(self, max_token_len, vocab,debug=False):
        super().__init__(debug)

        self.max_token_len = max_token_len
        self.word_vocab = vocab["word_vocab"]
        self.word2id= self.word_vocab.get_label2id_dict()
        self.vocab = vocab
        self.unk_id,self.eos_id,self.pad_id = self.word2id["<unk>"],self.word2id["<eos>"],self.word2id["<pad>"]
        self.line2id = lambda line: list(map(lambda word: self.word2id.get(word, self.unk_id), line))[:max_token_len-1]


    def _process(self, data):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")
        for sample in tqdm(data["words"]):
            word_ids = self.line2id(sample)
            label_ids = word_ids[1:] + [self.eos_id]
            word_length = len(word_ids)
            word_ids += [self.pad_id] * (self.max_token_len  - word_length)
            label_ids += [self.pad_id] * (self.max_token_len  - word_length)
            key_padding_mask = [not (id == self.pad_id) for id in word_ids]
            src_mask = generate_square_subsequent_mask(self.max_token_len).tolist()
            datable("word_ids",word_ids)
            datable("label_ids",label_ids)
            datable("word_length",word_length)
            datable("key_padding_mask",key_padding_mask)
            datable("src_mask",src_mask)
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return self._process(data)

def generate_square_subsequent_mask(sz: int) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == "__main__":

    cache_file = "/data/hongbang/projects/PatentClassification/datapath/language_models/chinese_news/cache/reader_datas.pkl"
    from cogtemplate.utils.io_utils import load_pickle,save_pickle

    train_data, dev_data, test_data, vocab = load_pickle(cache_file)

    processor = ChineseNewsForLMProcessor( max_token_len=128, vocab=vocab,debug=True)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    save_pickle([train_dataset, dev_dataset, test_dataset, vocab],"/data/hongbang/projects/PatentClassification/datapath/language_models/chinese_news/cache/processor_datas_debug.pkl")
    print("end")
