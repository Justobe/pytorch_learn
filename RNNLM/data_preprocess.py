


def get_data_list(filename):
    #filename = "Hamlet.txt"
    symbol_list = [',', '.', '!', '?', '-', ':', ';', '\n']
    symbol = '\''
    f = open(filename,"r+")
    all_data = f.read()
    for chr in symbol_list:
        all_data = all_data.replace(chr, " ")
    tmp_list = all_data.split(" ")
    symbol_list.append(symbol)
    res_list = []
    for tmp_chr in tmp_list:
        if tmp_chr!=' ' and tmp_chr!='':
            for chr in symbol_list:
                tmp_chr.strip(chr)
            res_list.append(tmp_chr)
    return res_list


#word_set = set(read_data(filename))
#print(word_set)


