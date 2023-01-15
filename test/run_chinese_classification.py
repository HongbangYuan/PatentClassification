from cogtemplate.data.readers.news_reader import NewsReader
from cogtemplate.data.processors.chinese_news_processors.news_processor_for_classification import NewsForClassificationProcessor
from cogtemplate import *
from cogtemplate.data.readers.patent_reader import PatentReader
from cogtemplate.data.processors.patent_processors.patent_base_processor import PatentForBertProcessor
from cogtemplate.core.predictor import Predictor
import torch.nn as nn
import torch.optim as optim

device, output_path = init_cogtemplate(
    device_id=7,
    # seed=66, # 0.506
    seed=0,
    output_path="/data/hongbang/projects/PatentClassification/datapath/text_classification/news/experimental_result",
    folder_tag="debug",
)

reader = NewsReader(
    raw_data_path="/data/hongbang/projects/PatentClassification/datapath/text_classification/news/raw_data")

train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

plm_name = "bert-base-chinese"
processor = NewsForClassificationProcessor(plm_name=plm_name, max_token_len=512, vocab=vocab, debug=False)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

plm = PlmAutoModel(pretrained_model_name=plm_name)
# plm._plm.resize_token_embeddings(len(processor.tokenizer))
model = BaseTextClassificationModel(plm=plm, vocab=vocab)
metric = BaseClassificationMetric(mode="multi")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)

# 新加了注释
trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=8,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=None,
                  metrics=metric,
                  train_sampler=None,
                  dev_sampler=None,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  num_workers=5,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=300,
                  save_steps=None,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  )
trainer.train()
print("end")