from .TfidfFormatter import TfidfFormatter


formatter_list = {
    "TfidfFormatter": TfidfFormatter,
}


def get_formatter(formatter_name,conf):
    if formatter_name in formatter_list.keys():
        return formatter_list[formatter_name](conf)
    else:
        raise NotImplementedError