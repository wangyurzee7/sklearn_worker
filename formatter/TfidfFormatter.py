import numpy as np
import os
import json
import random
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from sklearn.externals import joblib

class TfidfFormatter:
    def __init__(self,conf):
        self.text_use_which=conf["text_use_which"]
        self.label_use_which=conf["label_use_which"]
        self.segmentor=jieba
        self.vectorizer=TfidfVectorizer(min_df=2, max_df=1.0, token_pattern='\\b\\w+\\b')
        self.task=conf["task"]
        if self.task=="Classification":
            self.label2id={}
            self.id2label=[]
    def format(self,data,train=False):
        texts=[]
        labels=[]
        meta_infos=[]
        for d in data:
            text=' '.join(self.segmentor.cut(d[self.text_use_which]))
            texts.append(text)
            if (labels is not None):
                if self.label_use_which in d:
                    label=d[self.label_use_which]
                    labels.append(label)
                else:
                    assert not train
                    labels=None
            if "meta_info" in d.keys():
                meta_infos.append(d["meta_info"])
            else:
                meta_infos.append({})
        
        if train:
            self.vectorizer.fit(texts)
            if self.task=="Classification":
                all_labels=list(set(labels))
                for l in all_labels:
                    if not l in self.label2id:
                        self.label2id[l]=len(self.label2id)
                self.id2label=all_labels
                labels=[self.label2id[l] for l in labels]
        
        texts=self.vectorizer.transform(texts)
        
        ret={
            "x":texts,
            "meta_info":meta_infos
        }
        if labels is not None:
            ret["y"]=labels
        return ret
    def pred2label(self,pred):
        if self.task!="Classification":
            return pred
        ret=[]
        for ele in pred:
            ret.append(self.id2label[ele])
        return ret
    def dump(self,path):
        if self.task=="Classification":
            json.dump(self.label2id,open(os.path.join(path,"label2id.json"),"w"))
            json.dump(self.id2label,open(os.path.join(path,"id2label.json"),"w"))
        joblib.dump(self.vectorizer,os.path.join(path,"tf-idf.m"))
    def load(self,path):
        if self.task=="Classification":
            self.label2id=json.load(open(os.path.join(path,"label2id.json"),"r"))
            self.id2label=json.load(open(os.path.join(path,"id2label.json"),"r"))
        self.vectorizer=joblib.load(os.path.join(path,"tf-idf.m"))

