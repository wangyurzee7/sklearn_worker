import os
import warnings

def top_ten_words(formatter, model):
    warnings.filterwarnings("ignore") # make output pretty
    coef=model.model.coef_.A
    vocabs=formatter.vectorizer.get_feature_names()
    n=len(vocabs)
    for c in coef:
        cur=list(zip(abs(c),range(n)))
        cur.sort(reverse=True)
        res=[]
        for i in range(min(n,10)):
            res.append((vocabs[cur[i][1]],round(cur[i][0],1)))
        print(res)
    

detail_printer_list = {
    "TopTenWords": top_ten_words,
}


def get_detail_printer(detail_printer_name):
    if detail_printer_name in detail_printer_list.keys():
        return detail_printer_list[detail_printer_name]
    else:
        raise NotImplementedError
