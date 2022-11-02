from cogtemplate import *
from cogtemplate.models.base_language_model import TransformerForLM,TransformerModel
from cogtemplate.core.metric.base_lm_metric import BaseLanguageModelMetric
import torch.nn as nn
import torch.optim as optim

cache_file = "/data/hongbang/projects/PatentClassification/datapath/language_models/chinese_news/cache/processor_datas.pkl"
train_dataset, dev_dataset, test_dataset,vocab = load_pickle(cache_file)

device, output_path = init_cogtemplate(
    device_id=7,
    output_path="/data/hongbang/CogAGENT/datapath/language_models/chinese_news/experimental_result",
    folder_tag="run_transformer_lm",
)


ntokens = len(vocab["word_vocab"])  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerForLM(ntokens, emsize, nhead, d_hid, nlayers, dropout)

loss = nn.CrossEntropyLoss(ignore_index=0) # ignore_index = pad_id
metric= BaseLanguageModelMetric()
optimizer = optim.Adam(model.parameters(), lr=0.004)

# 新加了注释
trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=128,
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
                  validate_steps=600,
                  # save_by_metric="F1",
                  metric_mode='min',
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





