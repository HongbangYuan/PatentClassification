import torch.nn as nn
import torch.optim as optim
import torch.random

from cogtemplate import *
from cogtemplate.data.readers.patent_reader import PatentReader
from cogtemplate.data.processors.patent_processors.patent_base_processor import PatentForBertProcessor


device, output_path = init_cogtemplate(
    device_id=9,
    # seed=66, # 0.506
    # seed=55,
    output_path="/data/hongbang/CogAGENT/datapath/text_classification/patent/experimental_result",
    folder_tag="simple_test",
)

reader = PatentReader(raw_data_path="/data/hongbang/CogAGENT/datapath/text_classification/patent/raw_data")
# train_data = reader._read_train()
train_data, dev_data ,test_data = reader.read_all()
vocab = reader.read_vocab()
# "hfl/chinese-bert-wwm"
# "bert-base-chinese"
# "hfl/chinese-pert-base"
# plm_name = "hfl/chinese-pert-base"
plm_name = 'hfl/chinese-roberta-wwm-ext'
processor = PatentForBertProcessor(plm=plm_name, max_token_len=512, vocab=vocab)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)

plm = PlmAutoModel(pretrained_model_name=plm_name)
model = BaseTextClassificationModel(plm=plm, vocab=vocab)
metric = BaseClassificationMetric(mode="multi")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[20,25],gamma=0.1)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[15,20],gamma=0.1)

trainer = Trainer(model,
                  dev_data=dev_dataset,
                  train_data=train_dataset,
                  n_epochs=35,
                  # n_epochs=17,
                  batch_size=32,
                  loss=loss,
                  optimizer=optimizer,
                  # scheduler=scheduler,
                  # scheduler_steps=27,
                  scheduler=None,
                  metrics=metric,
                  train_sampler=None,
                  dev_sampler=None,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  num_workers=5,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=20,
                  save_by_metric="macro_F1",
                  save_steps=None,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  )
trainer.train()
