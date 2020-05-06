import json

class JsonLoader:
    def __init__(self,conf):
        pass
    def load(self,file_name):
        return json.load(open(file_name,"r"))

