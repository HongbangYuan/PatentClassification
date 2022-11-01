import os
import numpy as np

root_path = "/data/hongbang/projects/PatentClassification/datapath/language_models/chinese_news/raw_data"
target_file = os.path.join(root_path,'news.2021.zh.shuffled.deduped')
train_file = os.path.join(root_path,"train.txt")
dev_file = os.path.join(root_path,"dev.txt")
test_file = os.path.join(root_path,"test.txt")
split = [8,1,1]


split = split / np.sum(np.array(split))

with open(target_file,"r") as f:
    lines = f.readlines()
    num = len(lines)
    divide_1 = int(split[0] * num)
    divide_2 = int((split[0] + split[1]) * num)

    train_lines = lines[ : divide_1]
    dev_lines = lines[divide_1 : divide_2]
    test_lines = lines[divide_2 : ]
print("Saving to files..")
for file,my_lines in zip([train_file,dev_file,test_file],[train_lines,dev_lines,test_lines]):
    with open(file,"w") as f:
        for items in my_lines:
            f.writelines(items)

print("Num Train:{}  Num Dev:{}  Num Test:{}".format(divide_1,divide_2 - divide_1,num - divide_2))
print("Preprocess Done!")
