import numpy as np
import os
import sys
import json
import random
import warnings
import argparse

from constants import CONST

from tools.run_tools import *
from tools.output_dumper import get_output_dumper

tails_need_removed=["conf","config","_"]
def guess_model_name(conf_name):
    temp=conf_name.split('/')[-1]
    if conf_name.find('.')!=-1:
        temp='.'.join(temp.split('.')[:-1])
    flag=True
    while flag:
        flag=False
        for tail in tails_need_removed:
            if temp.endswith(tail):
                temp=temp[:-len(tail)]
                flag=True
    return temp

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="Config file (Json format)", required=True)
    parser.add_argument('--train', help="Train?", action="store_true")
    parser.add_argument('--test', help="Test?", action="store_true")
    args = parser.parse_args()
    
    whether_train=args.train
    whether_test=args.test
    conf_file_name=args.config
    if not whether_train and not whether_test:
        whether_train=True
        whether_test=True

    conf=json.load(open(conf_file_name,"r"))
    
    if not os.path.exists(CONST["model_dumped_path"]):
        os.mkdir(CONST["model_dumped_path"])
    model_name=guess_model_name(conf_file_name)
    
    if "train_path_params" in conf:
        train_file_path_list={
            "{}_{}".format(model_name,path_param) : conf["train_file_path"].format(path_param) 
            for path_param in conf["train_path_params"]
        }
    else:
        train_file_path_list={model_name:conf["train_file_path"]}
    
    for name,path in train_file_path_list.items():
        formatter,model=train_and_valid(
            file_path=path,
            model_name=name,
            conf=conf,
            skip_training=not whether_train,
        )

        if whether_test:
            output_dumper=get_output_dumper(conf["test_output_dumper"])
            if "test_path_params" in conf:
                for path_param in conf["test_path_params"]:
                    test(
                        formatter=formatter,
                        model=model,
                        file_path=conf["test_file_path"].format(path_param),
                        model_name=name,
                        test_title="test | {} | {}".format(model_name,path_param),
                        conf=conf,
                        valid=False,
                        output_dumper=output_dumper,
                        out_file_name=path_param,
                    )
            else:
                test(
                    formatter=formatter,
                    model=model,
                    file_path=conf["test_file_path"],
                    model_name=name,
                    test_title="test | {}".format(model_name),
                    conf=conf,
                    valid=False,
                    output_dumper=output_dumper,
                )
        