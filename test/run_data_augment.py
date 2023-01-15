import torch.nn as nn
import torch.optim as optim
import torch.random

from cogtemplate import *
from cogtemplate.data.readers.patent_reader import PatentReader
from cogtemplate.data.processors.patent_processors.patent_base_processor import PatentForBertProcessor
from cogtemplate.core.predictor import Predictor
from cogtemplate.data.datable import DataTable
from cogtemplate.data.datableset import DataTableSet
from tqdm import tqdm

from nlpcda import Randomword

class Augmenter:
    def __init__(self):
        self.augment_num = 3
        self.smw = Randomword(create_num=self.augment_num,change_rate=0.6)

    def augment_one_line(self,sample,headers):
        dict_data = {key:value for value,key in zip(sample,headers)}
        augmented_abstract = self.smw.replace(dict_data['abstract'])
        new_dict_data = {}

        for key,value in dict_data.items():
            new_dict_data[key] = [value] * self.augment_num
            if key in ['abstract']:
                new_dict_data[key] = self.smw.replace(dict_data[key])
        print("Debug Usage")
        list_of_dict_data = []
        for i in range(self.augment_num):
            list_of_dict_data.append({k:v[i] for k,v in new_dict_data.items()})
        return list_of_dict_data.extend(dict_data)

    def augment(self,data):
        datable = DataTable()
        print("Processing data...")
        for idx in range(len(data)):
            single_sample = data[idx]
            augment_samples = self.augment_one_line(single_sample,data.headers)
            for sample in augment_samples:
                for key in ["id", "title", "assignee", "abstract"]:
                    datable(key, sample[key])
                if 'label_id' in sample:
                    datable("label",sample["label_id"])
        return datable

reader = PatentReader(raw_data_path="/data/hongbang/projects/PatentClassification/datapath/text_classification/patent/raw_data")
# train_data = reader._read_train()
train_data, dev_data ,test_data = reader.read_all(split=None)
vocab = reader.read_vocab()
augmenter = Augmenter()
train_data = augmenter.augment(train_data)


# choose one plm model
# plm_name = "hfl/chinese-bert-wwm"
# plm_name = "bert-base-chinese"
# plm_name = "hfl/chinese-pert-base"
plm_name = 'hfl/chinese-roberta-wwm-ext'
processor = PatentForBertProcessor(plm=plm_name, max_token_len=512, vocab=vocab)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)

plm = PlmAutoModel(pretrained_model_name=plm_name)
model = BaseTextClassificationModel(plm=plm, vocab=vocab)
metric = BaseClassificationMetric(mode="multi")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[20,25],gamma=0.1)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[15,20],gamma=0.1)
