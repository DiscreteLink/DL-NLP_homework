import jieba
import paddle
import paddle.nn as nn

# embed_size: 词嵌入后的特征数；
embed_size = 128
# hidden_size: lstm中隐层的节点数；
hidden_size = 1024
# num_layers: lstm中的隐层数量；
num_layers = 1
# num_epochs: 全文本遍历的次数；
num_epochs = 100
# batch_size: 全样本被拆分的batch组数量
batch_size = 30
# seq_length: 获取的序列长度
seq_length = 60
# learning_rate: 模型的学习率
learning_rate = 0.001


class Dictionary(object):
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1


class Corpus(object):
    
    def __init__(self):
        self.dictionary = Dictionary()

    
    def get_data(self, path, batch_size=20):
        # step 1 生成字典
        deletewords = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '新语丝电子文库',
                  '\u3000', '\n', '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
        with open(path, 'r', encoding="ANSI") as f:
            tokens = 0
            for line in f.readlines():
                for deleteword in deletewords:
                    line = line.replace(deleteword, '')
                words = jieba.lcut(line) + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # step 2 转化成向量
        ids = paddle.zeros([tokens])
        token = 0
        with open(path, 'r', encoding="ANSI") as f:
            for line in f.readlines():
                for deleteword in deletewords:
                    line = line.replace(deleteword, '')
                words = jieba.lcut(line) + ['<eos>']
                for word in words:
                    # 只对每一个词给出一个编号
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        # step 3 分成不同的batch
        num_batches = ids.shape[0] // batch_size
        # 这里是取了整的batch
        ids = ids[:num_batches * batch_size]
        ids=paddle.reshape(ids, [batch_size, -1])
        return ids
    



class LSTMmodel(nn.Layer):
    # vocab_size字典里面的个数
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        # embed: 通过nn.Embedding初始化一个词嵌入层，用来将映射的one-hot向量词向量化。
        # 输入的参数是映射表长度(vocab_size即单词总数)和词嵌入空间的维数(embed_size即每个单词的特征数)
        self.embed = nn.Embedding(vocab_size, embed_size)
        # lstm: 通过nn.LSTM初始化一个LSTM层，是整个模型最核心、也是唯一的隐藏层。
        # 输入的参数是词嵌入空间的维数(embed_size即每个单词的特征数)、隐藏层的节点数(即hidden_size)和隐藏层的数量(即num_layers)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, time_major=False)
        # 通过nn.Linear初始化一个全连接层，用来把神经网络的运算结果转化为单词的概率分布。
        # 输入的参数是LSTM隐藏层的节点数(即hidden_size)和所有单词的数量(即vocab_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    # 传入的参数是输入值矩阵x和上一次运算得到的参数矩阵h：
    def forward(self, x, h):
        x = self.embed(x)
        # 经过词嵌入后扩充了维度

        out, (h, c) = self.lstm(x, h)

        out = out.reshape([out.shape[0] * out.shape[1], out.shape[2]])
        
        out = self.linear(out)

        return out, (h, c)
    

corpus = Corpus()
ids = corpus.get_data('data/越女剑.txt', batch_size)
vocab_size = len(corpus.dictionary)
print("词典大小："+str(vocab_size))

model = LSTMmodel(vocab_size, embed_size, hidden_size, num_layers)
model_output = LSTMmodel(vocab_size, embed_size, hidden_size, num_layers)
cost = nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate)

print("第一个词：")
print(corpus.dictionary.idx2word[17])
print("")
for epoch in range(num_epochs):
    # states是参数矩阵的初始化，相当于对LSTMmodel类里的(h, c)的初始化；
    states = (paddle.zeros(shape=[num_layers, batch_size, hidden_size],dtype='float32'),
            paddle.zeros(shape=[num_layers, batch_size, hidden_size],dtype='float32'))
 
    for i in range(0, ids.shape[1] - seq_length, seq_length):
        inputs = ids[:, i:i+seq_length]
        inputs=paddle.cast(inputs, dtype='int64')
        # target始终取inputs的下一个字符
        targets = ids[:, (i+1):(i+1)+seq_length]
        targets=paddle.cast(targets, dtype='int64')

        states = [state.detach() for state in states]
        # 把inputs和states传入model，得到通过模型计算出来的outputs和更新后的states
        outputs, states = model(inputs, states)
        loss = cost(outputs, targets.reshape([-1]))

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        print(f'\r epoch:{epoch}/{num_epochs} process:{i//seq_length}/{ids.shape[1]//seq_length} loss:{loss.item():.3f}',end='')
    print()
    paddle.save(model.state_dict(), "model/test_model.pdparams")
    pdparams = paddle.load("model/test_model.pdparams")
    model_output.set_state_dict(pdparams)
    num_samples = 30
    article = str()
    state_output = (paddle.zeros(shape=[num_layers, 1, hidden_size],dtype='float32'),
        paddle.zeros(shape=[num_layers, 1, hidden_size],dtype='float32'))
    _input = paddle.to_tensor([17]).unsqueeze(1)
    for i in range(num_samples):
        output, state_output = model(_input, state_output)
        # prob是对上一步得到的output进行指数化，加强高概率结果的权重；
        prob = output.exp()
        # word_id，通过torch_multinomial，以prob为权重，对结果进行加权抽样，样本数为1(即num_samples)
        word_id = paddle.multinomial(prob, num_samples=1).item()
        # 为下一次运算作准备，通过fill_方法，把最新的结果(word_id)作为_input的值
        _input.fill_(word_id)
        # 从字典映射表Dictionary里，找到当前索引(即word_id)对应的单词；
        word = corpus.dictionary.idx2word[word_id]
        article += word
    print(article)
    
