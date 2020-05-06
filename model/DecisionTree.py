from sklearn import tree
import os
from sklearn.externals import joblib

def set_model_params(model,params):
    for key in params.keys():
        val=params[key]
        # if key=="C":
        #     model.set_params(C=val)
        # elif key=="kernel":
        #     model.set_params(kernel=val)
        # else:
        #     pass

class DecisionTree:
    def __init__(self,conf):
        if conf["task"]=="Classification":
            self.model=tree.DecisionTreeClassifier()
        elif conf["task"]=="Regression":
            self.model=tree.DecisionTreeRegressor()
        else:
            pass
        set_model_params(self.model,conf["model_params"])
    def fit(self,data):
        self.model.fit(data["x"],data["y"])
    def predict(self,data):
        return self.model.predict(data["x"])
    def dump(self,path):
        joblib.dump(self.model,os.path.join(path,"model.m"))
    def load(self,path):
        self.model=joblib.load(os.path.join(path,"model.m"))
        
