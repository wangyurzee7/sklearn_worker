import pandas as pd

class CsvLoader:
    def __init__(self,conf):
        self.sep=conf["csv_separator"]

    def load(self,file_name):
        data=pd.read_csv(file_name, sep=self.sep)
        obj=data.to_dict()
        ret={}
        for k in obj.keys():
            for i in obj[k].keys():
                if i not in ret:
                    ret[i]={}
                ret[i][k]=obj[k][i]
        ret=list(ret.values())
        # print(ret)
        return ret