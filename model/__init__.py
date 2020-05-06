from .SVM import SVM

model_list = {
    "SVM": SVM,
}


def get_model(model_name,conf):
    if model_name in model_list.keys():
        return model_list[model_name](conf)
    else:
        raise NotImplementedError
