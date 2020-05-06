from .SVM import SVM
from .DecisionTree import DecisionTree

model_list = {
    "SVM": SVM,
    "DecisionTree": DecisionTree,
}


def get_model(model_name,conf):
    if model_name in model_list.keys():
        return model_list[model_name](conf)
    else:
        raise NotImplementedError
