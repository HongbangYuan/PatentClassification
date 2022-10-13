from cogtemplate.data.datable import DataTable
from cogtemplate.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogtemplate.data.processors.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()  # set transformers logging level


class PatentForBertProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab,debug=False):
        super().__init__(debug)
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def _process(self, data,is_training=True):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")
        pbar = tqdm(zip(data['abstract'],data['title'], data['label'],data['id']),total=len(data["title"])) \
            if is_training else tqdm(zip(data['abstract'],data['title'], data['id']),total=len(data["title"]))
        for sample in pbar:
            # title = process_title(title)
            if is_training:
                sentence, title, label, id = sample
            else:
                sentence,title,id = sample
            tokenized_data = self.tokenizer.encode_plus(text=title + sentence,
                                                        # text_pair=sentence,
                                                        padding="max_length",
                                                        truncation=True,
                                                        add_special_tokens=True,
                                                        max_length=self.max_token_len)
            datable("input_ids", tokenized_data["input_ids"])
            datable("token_type_ids", tokenized_data["token_type_ids"])
            datable("attention_mask", tokenized_data["attention_mask"])
            datable("id",id)
            if is_training:
                datable("label", self.vocab["label_vocab"].label2id(label))
        datable.not2torch.add("id")
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return self._process(data,is_training=False)

def process_title(title):
    for word in ['一种','一种','装置','设备']:
        title = title.replace(word,"")
    return title

if __name__ == "__main__":
    from cogtemplate.data.readers.patent_reader import PatentReader

    reader = PatentReader(raw_data_path="/data/hongbang/CogAGENT/datapath/text_classification/patent/raw_data")
    # train_data = reader._read_train()
    train_data, dev_data = reader.read_all()
    vocab = reader.read_vocab()

    print("Processing data...")
    processor = PatentForBertProcessor(plm="bert-base-chinese", max_token_len=512, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    print("end")
