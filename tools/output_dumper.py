import json

def json_dumper(origin_data,pred,file_prefix):
    file_name=file_prefix+".json"
    out=[]
    for d,p in zip(origin_data,pred):
        try:
            json.dumps(p)
            d["output"]=p
        except:
            d["output"]=str(p)
        out.append(d)
    json.dump(out,open(file_name,"w"),indent=2,ensure_ascii=False)


def csv_dumper(origin_data,pred,file_prefix):
    file_name=file_prefix+".csv"
    with open(file_name,"w") as f:
        f.write("id,predicted\n")
        for i,p in enumerate(pred):
            f.write("{},{}\n".format(i+1,p))



output_dumper_list = {
    "JsonFormatOutputDumper": json_dumper,
    "CsvFormatOutputDumper": csv_dumper,
}


def get_output_dumper(output_dumper_name):
    if output_dumper_name in output_dumper_list.keys():
        return output_dumper_list[output_dumper_name]
    else:
        raise NotImplementedError
