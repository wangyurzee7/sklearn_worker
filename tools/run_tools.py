import numpy as np
import os
import json
import random
import warnings

from formatter.__init__ import get_formatter
from model.__init__ import get_model
from dataset.__init__ import get_dataset_loader
from tools.accuracy import get_accuracy
from tools.detail_printer import get_detail_printer

from constants import CONST


def format_datas(file_list,formatter,conf,shuffle=False,train=False,return_origin_data=False):
    max_n=CONST["max_data_num"]
    assert (not shuffle) or train
    data_loader=get_dataset_loader(conf["dataset_format"],conf)
    actual_n=sum([len(data_loader.load(f)) for f in file_list])
    ratio=1 if actual_n<max_n else max_n/actual_n
    arr=[]
    for f in file_list:
        cur_arr=data_loader.load(f)
        if shuffle:
            random.shuffle(cur_arr)
        cur_arr=cur_arr[:round(ratio*len(cur_arr))]
        arr.extend(cur_arr)
    if shuffle:
        random.shuffle(arr)
    ret=formatter.format(arr,train=train)
    if return_origin_data:
        return ret,arr
    else:
        return ret

def dump_model(model_name,model,formatter):
    dst_path=os.path.join(CONST["model_dumped_path"],model_name)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    formatter.dump(dst_path)
    model.dump(dst_path)

def load_model(model_name,model,formatter):
    src_path=os.path.join(CONST["model_dumped_path"],model_name)
    formatter.load(src_path)
    model.load(src_path)

def test(formatter,model,file_path,model_name,test_title,conf,out_file_name=None,valid=False,output_dumper=None):
    file_list=conf["valid_file_list"] if valid else conf["test_file_list"]
    test_file_list=[os.path.join(file_path,f) for f in file_list]
    if output_dumper is None:
        data=format_datas(test_file_list,formatter,conf)
    else:
        data,origin_data=format_datas(test_file_list,formatter,conf,return_origin_data=True)
    result=model.predict(data)
    if "pred2label" in dir(formatter):
        result=formatter.pred2label(result)
    
    accuracy_methods={acc:get_accuracy(acc) for acc in conf["accuracy_methods"]}
    pretty_result={}
    assert (not valid) or ("y" in data)
    if "y" in data:
        if "pred2label" in dir(formatter):
            data["y"]=formatter.pred2label(data["y"])
        for acc in accuracy_methods.keys():
            pretty_result[acc]=accuracy_methods[acc](data["y"],result)
        print("[ {} ] {}".format(test_title,str(pretty_result)))
        
        if ("save_log" in conf) and conf["save_log"]:
            if out_file_name is not None:
                log_file_name=os.path.join(CONST["model_dumped_path"],model_name,"{}.log.txt".format(out_file_name))
            else:
                log_file_name=os.path.join(CONST["model_dumped_path"],model_name,"log.txt")
            dumped=list(zip(data["meta_info"],result,data["y"]))
            json.dump(dumped,open(log_file_name,"w"),ensure_ascii=False)
    else:
        print("[ {} ] Test Completed.".format(test_title))
    
    if output_dumper is not None:
        if out_file_name is not None:
            result_file_prefix=os.path.join(CONST["model_dumped_path"],model_name,"{}.{}".format(conf["test_output_dumper"],out_file_name))
        else:
            result_file_prefix=os.path.join(CONST["model_dumped_path"],model_name,"{}".format(conf["test_output_dumper"]))
        output_dumper(origin_data,result,result_file_prefix)

def train_and_valid(file_path,model_name,conf,skip_training=False):
    formatter=get_formatter(conf["formatter"],conf)
    model=get_model(conf["model"],conf)
    
    if skip_training:
        load_model(model_name,model,formatter)
    else:
        train_file_list=[os.path.join(file_path,f) for f in conf["train_file_list"]]
        data=format_datas(train_file_list,formatter,conf,shuffle=conf["shuffle"],train=True)
        model.fit(data)
        dump_model(model_name,model,formatter)
    
    if "detail_printer" in conf:
        for printer_name in conf["detail_printer"]:
            get_detail_printer(printer_name)(formatter,model)
    
    if "valid_path_params" in conf:
        for path_param in conf["valid_path_params"]:
            test(
                formatter=formatter,
                model=model,
                file_path=conf["valid_file_path"].format(path_param),
                model_name=model_name,
                test_title="valid | {} | {}".format(model_name,path_param),
                conf=conf,
                valid=True,
                out_file_name=path_param,
            )
    else:
        test(
            formatter=formatter,
            model=model,
            file_path=conf["valid_file_path"],
            model_name=model_name,
            test_title="valid | {}".format(model_name),
            conf=conf,
            valid=True,
        )
    return formatter,model


