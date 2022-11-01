import json
import csv
from tqdm import tqdm
# from eda import *
from copy import deepcopy
from nlpcda import Similarword

input_file = "/data/hongbang/projects/PatentClassification/datapath/text_classification/patent/raw_data/train.json"
output_json_file = "/data/hongbang/projects/PatentClassification/datapath/text_classification/patent/raw_data/augmented_train.json"


with open(input_file, 'r', encoding='utf8') as file:
    lines = file.readlines()
    all_dict_data = [json.loads(line) for line in lines]

smw = Similarword(create_num=3,change_rate=0.5)
augmented_dict_data = []

replace_key = 'abstract'
for dict_data in tqdm(all_dict_data):
    sentence = dict_data[replace_key]

    # 相似字替换
    augment_sentences = smw.replace(sentence)
    for new_sentence in augment_sentences:
        new_dict_data = deepcopy(dict_data)
        new_dict_data[replace_key] = new_sentence
        augmented_dict_data.append(new_dict_data)

fieldnames = [key for key in all_dict_data[0].keys()]

# alpha = 0.05
# num_aug = 8
# aug_sentences = eda("sentence", alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)

with open(output_json_file,'w') as f:
    for dic in augmented_dict_data:
        json.dump(dic, f,ensure_ascii=False)
        f.write("\n")

# with open(output_csv_file,"w") as f:
#     writer = csv.DictWriter(f, delimiter='\t',fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(augmented_dict_data)
print("Original Length:{} and augmented length:{}".format(len(all_dict_data),len(augmented_dict_data)))
print("Write results to file {}".format(output_json_file))