import matplotlib.pyplot as plt
import jieba
import os
import random
import gensim
from gensim.models.ldamodel import LdaModel
from sklearn.svm import SVC


def read_data(file_names):
    paras_list = []
    labels_list = []
    for index, file_name in enumerate(file_names):
        file_path = os.path.join(sub_folder, file_name+'.txt')
        with open(file_path, 'r', encoding='ANSI') as f:
            data = f.read()
            f.close()
        for paragraph in data.split('\n'):
            if(len(paragraph) < 500):
                continue
            paras_list.append(paragraph)
            labels_list.append(index)
    return paras_list, labels_list


def load_delete_words():
    delete = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '新语丝电子文库',
              '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、',
              '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    with open('./cn_stopwords.txt', 'r', encoding='utf-8') as f:
        for a in f:
            if a != '\n':
                delete.append(a.strip())
    return delete


# 读取源文件
base_path = os.getcwd()
sub_folder = os.path.join(base_path, 'corpus')
inf_txt_path = os.path.join(sub_folder, 'inf.txt')
with open(inf_txt_path, 'r') as f:
    content = f.read()
file_names = content.split(",")
paras_list, labels_list = read_data(file_names)

# 均匀抽取200个段落
chosen_paras = []
chosen_labels = []
random.seed(0)  # 保证每次抽到的200个段落保持一致
random_indices = random.sample(range(len(paras_list)), 200)
for i in random_indices:
    chosen_paras.append(paras_list[i])
    chosen_labels.append(labels_list[i])

# 完成处理
deletewords = load_delete_words()
for i in range(200):
    for deleteword in deletewords:
        chosen_paras[i] = chosen_paras[i].replace(deleteword, '')

# 分词，分别以字和词为基本单位
tokens_word = []  # 以词为单位
tokens_word_label = []
tokens_char = []  # 以字为单位
tokens_char_label = []
for i, text in enumerate(chosen_paras):
    words = list(jieba.cut(text))
    tokens_word.append(words)
    tokens_word_label.append(chosen_labels[i])

    temp = []
    for word in words:
        temp.extend([char for char in word])
    tokens_char.append(temp)
    tokens_char_label.append(chosen_labels[i])

# 划分训练集与测试集
train_p = 0.7
trainset_size_word = int(len(tokens_word_label)*train_p)  # 根据词划分
label_train_word = tokens_word_label[0:trainset_size_word]
label_test_word = tokens_word_label[trainset_size_word:]
text_train_word = tokens_word[0:trainset_size_word]
text_test_word = tokens_word[trainset_size_word:]
trainset_size_char = int(len(tokens_char_label)*train_p)  # 根据字划分
label_train_char = tokens_char_label[0:trainset_size_char]
label_test_char = tokens_char_label[trainset_size_char:]
text_train_char = tokens_char[0:trainset_size_char]
text_test_char = tokens_char[trainset_size_char:]

# 构造词典
dictionary_word = gensim.corpora.Dictionary(tokens_word)  # 以词为单位
dictionary_char = gensim.corpora.Dictionary(tokens_char)  # 以字为单位

# 基于词典，构造文本向量
corpus_train_word = [dictionary_word.doc2bow(
    tokens) for tokens in text_train_word]
corpus_test_word = [dictionary_word.doc2bow(
    tokens) for tokens in text_test_word]
corpus_train_char = [dictionary_char.doc2bow(
    tokens) for tokens in text_train_char]
corpus_test_char = [dictionary_char.doc2bow(
    tokens) for tokens in text_test_char]


if __name__ == '__main__':
    accuracy_test_word = []
    accuracy_test_char = []
    l1 = []
    l2 = []
    x = range(1, 501, 10)
    for j in x:
        feature_train = []
        feature_test = []
        lda_word = LdaModel(corpus=corpus_train_word,
                            id2word=dictionary_word, num_topics=j)
        topics_train = lda_word.get_document_topics(
            corpus_train_word, minimum_probability=0)
        topics_test = lda_word.get_document_topics(
            corpus_test_word, minimum_probability=0)
        for i in range(0, len(topics_train)):
            feature_train.append([k[1] for k in topics_train[i]])
        for i in range(0, len(topics_test)):
            feature_test.append([k[1] for k in topics_test[i]])
        # 定义SVM分类器
        clf = SVC(kernel='rbf', decision_function_shape='ovr')
        # 训练模型
        clf.fit(feature_train, label_train_word)
        accuracy_test_word.append(clf.score(feature_test, label_test_word))
        l1.append(clf.score(feature_train, label_train_word))

        feature_train = []
        feature_test = []
        lda_char = LdaModel(corpus=corpus_train_char,
                            id2word=dictionary_char, num_topics=j)
        topics_train = lda_char.get_document_topics(
            corpus_train_char, minimum_probability=0)
        topics_test = lda_char.get_document_topics(
            corpus_test_char, minimum_probability=0)
        for i in range(0, len(topics_train)):
            feature_train.append([k[1] for k in topics_train[i]])
        for i in range(0, len(topics_test)):
            feature_test.append([k[1] for k in topics_test[i]])
        # 定义SVM分类器
        clf = SVC(kernel='rbf', decision_function_shape='ovr')
        # 训练模型
        clf.fit(feature_train, label_train_char)
        accuracy_test_char.append(clf.score(feature_test, label_test_char))
        l2.append(clf.score(feature_train, label_train_char))

    t = x
    plt.plot(t, accuracy_test_word, label='word')
    plt.plot(t, accuracy_test_char, label='char')
    plt.xlabel('主题数目')
    plt.ylabel('准确率')
    plt.title('主题-预测准确率变化情况-测试集')
    plt.legend()
    plt.show()

    t = x
    plt.plot(t, l1, label='word')
    plt.plot(t, l2, label='char')
    plt.xlabel('主题数目')
    plt.ylabel('准确率')
    plt.title('主题-预测准确率变化情况-训练集')
    plt.legend()
    plt.show()
