from cogtemplate.data.datable import DataTable
from cogtemplate.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogtemplate.data.processors.base_processor import BaseProcessor
from torch import Tensor
import torch
transformers.logging.set_verbosity_error()  # set transformers logging level
from transformers import AutoTokenizer

class NewsForClassificationProcessor(BaseProcessor):
    def __init__(self, plm_name, max_token_len, vocab,debug=False):
        super().__init__(debug)

        self.max_token_len = max_token_len
        self.label_vocab = vocab["label_vocab"]
        self.tokenizer = AutoTokenizer.from_pretrained(plm_name)


    def _process(self, data):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")
        for sample in tqdm(data):
            label,sentence = sample
            input_batch = self.tokenizer.encode_plus(sentence.replace(" ",""),padding='max_length',add_special_tokens=True,max_length=self.max_token_len,truncation=True)
            label_id = self.label_vocab.label2id(label)
            datable("label",label_id)
            datable('input_ids',input_batch["input_ids"])
            datable('attention_mask',input_batch["attention_mask"])
            datable('token_type_ids',input_batch['token_type_ids'])

        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return self._process(data)


if __name__ == "__main__":
    from cogtemplate.data.readers.news_reader import NewsReader
    reader = NewsReader(raw_data_path="/data/hongbang/projects/PatentClassification/datapath/text_classification/news/raw_data")

    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = NewsForClassificationProcessor(plm_name='bert-base-uncased',max_token_len=512, vocab=vocab,debug=True)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=train_dataset, batch_size=8, )
    sample = next(iter(dataloader))