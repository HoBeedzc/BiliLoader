"""
对简单实现的bm25f算法的封装
"""
import json
import jieba
from functools import reduce
import os

stopwords_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "Chinese_stopwords.txt")


def stopwordslist(stopwords_path):  # 根据停用词文件路径生成停用词列表
    stopwords = [
        line.strip()
        for line in open(stopwords_path, 'r', encoding='utf-8').readlines()
    ]
    return stopwords


def json_load(filepath):  # 将数据从json文件转为字典
    with open(filepath, encoding='utf-8', errors='ignore') as f:
        dic = json.load(f)
    return dic


# BM25F核心算法部分
def query_analysis(query,
                   stopwords_path=stopwords_path):  # 对query进行处理，转换为对应的分词结果
    query = query.replace(' ', '')
    query = jieba.lcut(query)
    stopwords = stopwordslist(stopwords_path)
    query = [w for w in query if w not in stopwords]
    return query


def RSV(bv, term_list, k1, b, inverted_index, v1, v2, avdl,
        idf_dict):  # 输入文档名，检索词列表、公式参数，计算其RSV得分
    def filter_bv(l):
        for tup in l:
            if tup[0] == bv:
                return tup
        return 0

    rsv = 0
    for term in term_list:
        a1 = idf_dict[term]
        term_info = filter_bv(inverted_index[term])
        if term_info:
            tf = v1 * term_info[1] + v2 * term_info[2]
            dl = v1 * term_info[3] + v2 * term_info[4]
        else:
            tf = 0
            dl = 0
        a2 = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avdl) + tf)
        rsv += a1 * a2
    return rsv


def bm25f(query, inverted_index='inverted_index.json', v1=0.8, v2=0.2, idf_dic='idf_dic.json'):
    inverted_index = json_load(os.path.join(os.path.split(os.path.abspath(__file__))[0], inverted_index))
    idf_dic = json_load(os.path.join(os.path.split(os.path.abspath(__file__))[0], idf_dic))
    query = query_analysis(query)
    doc_list = []
    searched_query = []
    for term in query:
        try:
            doc_list.append(inverted_index[term])
            searched_query.append(term)
        except:
            print("没有包含关键词'%s'的文档!" % term)
    if len(doc_list) == 0:
        print("没有检索到包含搜索关键词的视频!")
        return
    l = []
    avdl = 0
    num = 0
    for term_doc in doc_list:
        l.append([doc[0] for doc in term_doc])
        for doc in term_doc:
            avdl += v1 * doc[3] + v2 * doc[4]
            num += 1
    avdl = avdl / num
    all_doc = reduce(lambda x, y: set(x) | set(y), l)  # 得到有关键词的所有文档集合
    score_dict = {}
    for doc in all_doc:
        score_dict[doc] = RSV(doc, searched_query, 0.1, 0.5, inverted_index,
                              v1, v2, avdl, idf_dic)
    score_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    return score_dict


def main():
    f1 = 'idf_dic.json'
    f2 = 'inverted_index.json'
    while 1:
        query = input()
        bm25f(query, f2, 0.8, 0.2, f1)


if __name__ == "__main__":
    main()
