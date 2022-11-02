from cogtemplate.core.metric.base_metric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch.nn as nn
import numpy as np

class BaseLanguageModelMetric(BaseMetric):
    def __init__(self, default_metric_name="ppl"):
        super().__init__()

        self.label_list = list()
        self.pre_list = list()
        self.default_metric_name = default_metric_name
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0) # ignore_index = pad_id
        self.val_loss = []

    def evaluate(self, pred, label):
        curr_loss = self.loss_function(pred,label)
        self.val_loss.append(curr_loss.item())

    def get_metric(self, reset=True):
        evaluate_result = {}
        evaluate_result["val_loss"] = np.mean(np.array(self.val_loss))
        evaluate_result["ppl"] = np.exp(np.mean(np.array(self.val_loss)))
        # if self.mode == "binary":
        #     P = precision_score(self.label_list, self.pre_list, average="binary")
        #     R = recall_score(self.label_list, self.pre_list, average="binary")
        #     F1 = f1_score(self.label_list, self.pre_list, average="binary")
        #     Acc = accuracy_score(self.label_list, self.pre_list)
        #     evaluate_result = {"P": P,
        #                        "R": R,
        #                        "F1": F1,
        #                        "Acc": Acc,
        #                        }
        # if self.mode == "multi":
        #     micro_P = precision_score(self.label_list, self.pre_list, average="micro")
        #     micro_R = recall_score(self.label_list, self.pre_list, average="micro")
        #     micro_F1 = f1_score(self.label_list, self.pre_list, average="micro")
        #     macro_P = precision_score(self.label_list, self.pre_list, average="macro")
        #     macro_R = recall_score(self.label_list, self.pre_list, average="macro")
        #     macro_F1 = f1_score(self.label_list, self.pre_list, average="macro")
        #     Acc = accuracy_score(self.label_list, self.pre_list)
        #     evaluate_result = {"micro_P": micro_P,
        #                        "micro_R": micro_R,
        #                        "micro_F1": micro_F1,
        #                        "macro_P": macro_P,
        #                        "macro_R": macro_R,
        #                        "macro_F1": macro_F1,
        #                        "Acc": Acc,
        #                        }
        if reset:
            self.label_list = list()
            self.pre_list = list()
        return evaluate_result