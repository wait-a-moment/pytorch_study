import torchtext
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import numpy as np
import random

"""
语言模型就是输入一个词预测下一个词是什么

1.torchtext数据处理
2.定义模型类
3.定义评估函数
4.初始化一个模型
5.定义Loss_fn 和optimizer
6.开始迭代训练
"""
USE_CUDA = torch.cuda.is_available()
# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

BATCH_SIZE = 32
EMBEDDING_SIZE = 32
MAX_VOCAB_SIZE = 5000

# 取训练数据的一小部分测试流程
# with open(r"D:\git_code\pytorch_study\data\text8\text8.train.txt","r",encoding="utf8") as f1:
#     line=f1.readline()
#     print(len(line))
#     with open(r"D:\git_code\pytorch_study\data\text8\text8.train_part.txt","a",encoding="utf8") as f2:
#         for i in line[:20000000]:
#             f2.write(i)
# 使用torchtext处理文本数据
TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path="./data/text8", train="text8.train_part.txt",
                                                                     validation="text8.dev.txt", test="text8.test.txt",
                                                                     text_field=TEXT)

TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
print("vocable size:{}".format(len(TEXT.vocab)))
VOCAB_SIZE = len(TEXT.vocab)

# 迭代器返回模型所需要的处理后的数据,迭代器主要分为Iterator，BucketIterator,BPTTIterator三种,基于BPTT(基于时间的反向传播算法)的迭代器，一般用于语言模型.
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=-1, bptt_len=32, repeat=False, shuffle=True)  # bptt_len表示每个句子长度

it = iter(test_iter)
batch = next(it)

# 查看数据
for i in range(5):
    batch = next(it)
    print(batch.text.data)
    print(batch.text[:, 0].data)
    print("data:" + " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 0].data]))
    print("target: " + " ".join([TEXT.vocab.itos[i] for i in batch.target[:, 0].data]))


# 定义模型
class RNNModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel,
              self).__init__()  # 这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。相当于把父类的init构造方法拿过来用, 并且可以对父类的__init__方法进行补充
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)  # 用于返回一个对象属性值 nn.LSTM 或 nn.GRU
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                             options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weight()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weight(self):
        initrange = 0.1
        # uniform_从均匀分布中抽样
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        # 线性层从hidden state转化为输出单词表 三维转二维[sequence_length,batch_size,embedding]-->[sequence_length*batch_size,embedding]
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2)))  # 通过decoder 每个字会得到一个embedding
        # 获得线性输出后 再变成三维 这里decoded.size(1)=output.size(2)
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        # LSTM  2个中间状态 hidden cell
        if self.rnn_type == 'LSTM':
            """
            ([nlayers,batch_size,nhid])
            ([2,32,650],[2,32,650])
            """
            return (weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad),
                    weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad))
        else:
            """
            [2,32,650]
            """
            return weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad)


model = RNNModel('LSTM', VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE, 2)

if USE_CUDA:
    model = model.cuda()


def evaluate(model, data):
    # eval()在测试之前加，否则有输入数据即使不训练，它也会改变权值
    # pytorch会自己把BatchNormalization和DropOut固定住，不会取平均，而是用训练好的值。
    model.eval()
    total_loss = 0.
    it = iter(data)
    total_count = 0.

    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += np.multiply(*data.size())
            total_loss += loss.item() * np.multiply(*data.size())

    loss = total_loss / total_count
    model.train()  # model.train()让model变成训练模式
    return loss


"""
我们需要定义下面的一个function，帮助我们把一个hidden state和计算图之前的历史分离。
"""


# Remove this part
# hidden ([2,32,650],[2,32,650]))
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history.复制值，但是和历史的联系被剪短了"""
    """
    #hidden ([2,32,650],[2,32,650]))
    """
    if isinstance(h, torch.Tensor):
        return h.detach()  # 返回一个新的从当前图中分离的 Variable
    else:
        return tuple(repackage_hidden(v) for v in h)


"""
定义Loss_fn 和optimizer
"""
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

val_losses = []
GRAD_CLIP = 1.
NUM_EPOCHS = 2
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        hidden = repackage_hidden(hidden)  # 使梯度只依赖于一次迭代所读取的小批量序列数据（防止梯度计算开销太大）

        model.zero_grad()
        output, hidden = model(data, hidden)  # 只有语言模型才会这么传，因为句子中字和字都是连着的，可以一直传下去，但是随着循环，计算图回越来越大，所以需要repackage_hidden

        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))  # batch_size*target_class_dim,batch_size
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        #
        if i % 1000 == 0:
            print("epoch", epoch, "iter", i, "loss", loss.item())

        if i % 10000 == 0:
            val_loss = evaluate(model, val_iter)

            if len(val_losses) == 0 or val_loss < min(val_losses):
                print("best model, val loss: ", val_loss)
                torch.save(model.state_dict(), "lm-best.th")
            # 如果loss降不下去，可以尝试调整学习率（可以自定义什么时候调整学习率，比如三次loss不降低就调整学习率）
            else:
                scheduler.step()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            val_losses.append(val_loss)

"""
加载模型  在验证集合 和测试集合上各跑一遍
先初始化一个模型
"""
best_model = RNNModel('LSTM', VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE, 2, dropout=0.5)
if USE_CUDA:
    best_model = best_model.cuda()
"""
加载模型参数 这是torch的推荐保存方式
"""
best_model.load_state_dict(torch.load("lm-best.th"))

"""
在验证集合  测试集合  上泡一下数据
"""
val_loss = evaluate(best_model, val_iter)
print('perplexity:', np.exp(val_loss))

test_loss = evaluate(best_model, test_iter)

"""
numpy.exp()：返回e的幂次方，e是一个常数为2.71828
"""
print("perplexity:test_loss:", np.exp(test_loss))

"""
使用训练好的模型生成一些句子。
"""
"""
([2,1,650],[2,1,650])这个产生一个隐藏状态

初始化个批次的数据
"""
hidden = best_model.init_hidden(1)  # 1 是batch_size
device = torch.device('cuda' if USE_CUDA else 'cpu')
"""
产生一个（1,1）的数字 作为输入  正常输入 【32,32】]1
1代表 bpttlen  1 代表批次
"""
input = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)
words = []
# 产生100个单词
for i in range(100):
    """
    input[1,1]  hidden([2,1,650],[2,1,650])
    output[1,1,50002] hidden ([2,1,650],[2,1,650])
    输入一个单词 预测一个单词 循环100次
    """
    output, hidden = best_model(input, hidden)
    # word_weights=[50002]  取最后一维度
    word_weights = output.squeeze().exp().cpu()
    # 按照权重 产生 50002维度 那个可能的值得索引  下标  也就是产生一个单词的索引
    # 拿到单词的idx 可以拿到单词
    # 按照权重产生单词 这样每次拿到的单词是不同的  增加多变性
    word_idx = torch.multinomial(word_weights, 1)[
        0]  # torch.multinomial(input, num_samples,replacement=False, out=None) → LongTensor 作用是对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标。
    # 作为下一次的输入
    input.fill_(word_idx)
    word = TEXT.vocab.itos[word_idx]
    words.append(word)

print(" ".join(words))
