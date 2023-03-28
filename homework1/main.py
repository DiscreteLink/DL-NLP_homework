import math
import time
import jieba
import os

def read_data(path):
    with open(path, 'r', encoding='ANSI') as f:
        data = f.read()
        f.close()

    delete = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '新语丝电子文库',
      '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、',
      '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']

    with open('./cn_stopwords.txt', 'r', encoding='utf-8') as f:
        for a in f:
            if a != '\n':
                delete.append(a.strip())

    for a in delete:
        data = data.replace(a, '')
        
    return data

def get_unigram_wf(words):
    dic = {}
    for i in range(len(words)):
        dic[words[i]] = dic.get(words[i], 0) + 1
    return dic

def get_bigram_wf(words):
    dic = {}
    for i in range(len(words) - 1):
        dic[(words[i], words[i + 1])] = dic.get((words[i], words[i + 1]), 0) + 1
    return dic

def get_trigram_wf(words):
    dic = {}
    for i in range(len(words) - 2):
        dic[((words[i], words[i + 1]), words[i + 2])] = dic.get(((words[i], words[i + 1]), words[i + 2]), 0) + 1
    return dic

def unigram_model(dic):
    begin = time.time()
    unigram_sum = sum([item[1] for item in dic.items()])
    entropy = 0
    for item in dic.items():
        entropy += -(item[1] / unigram_sum) * math.log(item[1] / unigram_sum, 2)
    entropy = round(entropy, 4)
    end = time.time()
    runtime = round(end - begin, 4)
    return entropy,runtime

def bigram_model(uni_dic,bi_dic):
    begin = time.time()
    bigram_num = sum([item[1] for item in bi_dic.items()])
    entropy = 0
    for bi_item in bi_dic.items():
        jp = bi_item[1] / bigram_num # 联合概率
        cp = bi_item[1] / uni_dic[bi_item[0][0]] # 条件概率
        entropy += -jp * math.log(cp, 2)
    entropy = round(entropy, 4)
    end = time.time()
    runtime = round(end - begin, 4)
    return entropy,runtime

def trigram_model(bi_dic,tri_dic):
    begin = time.time()
    trigram_num = sum([item[1] for item in tri_dic.items()])
    entropy = 0
    for tri_item in tri_dic.items():
        jp = tri_item[1] / trigram_num
        cp = tri_item[1] / bi_dic[tri_item[0][0]]
        entropy += -jp * math.log(cp, 2)
    entropy = round(entropy, 4)
    end = time.time()
    runtime = round(end - begin, 4)
    return entropy,runtime

base_path = os.getcwd()
sub_folder = os.path.join(base_path, 'corpus')
inf_txt_path = os.path.join(sub_folder, 'inf.txt')
with open(inf_txt_path, 'r') as f:
    content = f.read()
file_names = content.split(",")
file_names.append("全文")


for file_name in file_names:
    file_path = os.path.join(sub_folder, file_name+'.txt')
    data = read_data(file_path)
    '''
    # 保存处理后的文本
    with open('./'+str(file_name)+'预处理.txt','w',encoding='utf-8') as f:
        f.write(data)
        f.close()
    '''
    # 先进行字分割
    words = [c for c in data]
    uni_dic = get_unigram_wf(words)
    bi_dic = get_bigram_wf(words)
    tri_dic = get_trigram_wf(words)
    unigram_entropy, timing = unigram_model(uni_dic)
    s = file_name + " 按字分割 1-Gram 信息熵为" + str(unigram_entropy) +' 耗时：' + str(timing)
    print(s)
    bigram_entropy, timing = bigram_model(uni_dic,bi_dic)
    s = file_name + " 按字分割 2-Gram 信息熵为" + str(bigram_entropy) +' 耗时：' + str(timing)
    print(s)
    trigram_entropy, timing = trigram_model(bi_dic,tri_dic)
    s = file_name + " 按字分割 3-Gram 信息熵为" + str(trigram_entropy) +' 耗时：' + str(timing)
    print(s)
    # 再进行词分割
    words = list(jieba.cut(data))
    uni_dic = get_unigram_wf(words)
    bi_dic = get_bigram_wf(words)
    tri_dic = get_trigram_wf(words)
    unigram_entropy,timing = unigram_model(uni_dic)
    s = file_name + " 按词分割 1-Gram 信息熵为" + str(unigram_entropy) +' 耗时：' + str(timing)
    print(s)
    bigram_entropy, timing = bigram_model(uni_dic,bi_dic)
    s = file_name + " 按词分割 2-Gram 信息熵为" + str(bigram_entropy) +' 耗时：' + str(timing)
    print(s)
    trigram_entropy, timing = trigram_model(bi_dic,tri_dic)
    s = file_name + " 按词分割 3-Gram 信息熵为" + str(trigram_entropy) +' 耗时：' + str(timing)
    print(s)
    with open('./corpus/全文.txt','a',encoding='ANSI') as f:
        f.write(data)
        f.close()

