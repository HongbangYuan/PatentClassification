# Chinese news corpus for the ucas NLP course
Download the dataset from https://data.statmt.org/news-crawl/zh/.    
Put the raw data in path`datapath/language_models/chinese_news/raw_data`.   
Like the following form:
```angular2html
datapath
├─ language_models
│  ├─ chiense_news
│  │  ├─ raw_data
│  │  │  ├─ news.2021.zh.shuffled.deduped
```
After running the preprocessing script ```./preprocess.py```,
the data is put like the following form:
```angular2html
datapath
├─ language_models
│  ├─ chiense_news
│  │  ├─ raw_data
│  │  │  ├─ news.2021.zh.shuffled.deduped
│  │  │  ├─ train.txt
│  │  │  ├─ dev.txt
│  │  │  ├─ test.txt
```