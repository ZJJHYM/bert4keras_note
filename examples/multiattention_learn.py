'''
详解链接：https://zhuanlan.zhihu.com/p/231631291

'''
from keras import initializers
from keras import activations
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf


class MySelfAttention(Layer):

    def __init__(self, output_dim, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.kernel_initializer = kernel_initializer
        super(MySelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(3, input_shape[2], self.output_dim),
                                 initializer=self.kernel_initializer,
                                 trainable=True)

        self.built = True

    def call(self, x):
        q = K.dot(x, self.W[0])
        k = K.dot(x, self.W[1])
        v = K.dot(x, self.W[2])

        e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))
        e = e / (self.output_dim ** 0.5)
        e = K.softmax(e)

        o = K.batch_dot(e, v)
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


def sequence_masking(x, mask, mode=0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    mode: 如果是0，则直接乘以mask；
          如果是1，则在padding部分减去一个大正数。
    axis: 序列所在轴，默认为1；
    """
    if mask is None or mode not in [0, 1]:
        return x
    else:
        if axis is None:
            axis = 1
        if axis == -1:
            axis = K.ndim(x) - 1
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask) - axis + 1):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 0:
            return x * mask
        else:
            return x - (1 - mask) * 1e12


class MyMultiHeadAttention(Layer):
    def __init__(self,
                 heads,
                 head_size,
                 key_size=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 **kwargs
                 ):
        self.heads = heads # 多头的数目
        self.head_size = head_size # v的头的大小
        self.out_dim = heads * head_size # 最终的输出维度

        # 一般情况下和head_size是一样的 key_size的作用可以参考苏神的低秩分解：https://kexue.fm/archives/7325
        self.key_size = key_size or head_size # qk每个头的大小
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        super(MyMultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MyMultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(
            units=self.heads * self.key_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.heads * self.key_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs, mask=None, a_mask=None, **kwargs):
        # 如果是自注意力，那么q,k,v都是同一个，shape=(btz, seq_len, emb_dim)
        # 如果不是自注意力，则q:shape=shape=(btz, q_seq_len, emb_dim), k,v相同:shape=shape=(btz, k_seq_len, emb_dim)
        q, k, v = inputs[:3]
        q_mask, v_mask, n = None, None, 3
        if mask is not None:
            if mask[0] is not None:
                q_mask = K.cast(mask[0], K.floatx())
            if mask[2] is not None:
                v_mask = K.cast(mask[2], K.floatx())
        if a_mask:
            a_mask = inputs[n]
            n += 1

        # 经过全连接层将最后一维映射成需要的维度
        qw = self.q_dense(q) # shape=(btz, q_seq_len, self.heads * self.key_size)
        kw = self.k_dense(k) # shape=(btz, k_seq_len, self.heads * self.key_size)
        vw = self.v_dense(v) # shape=(btz, k_seq_len, self.heads * self.head_size)  注意v这里和q,k有点不同
        # 形状变换  拆成多头
        qw = K.reshape(qw, shape=(-1, K.shape(qw)[1], self.heads, self.key_size))  # shape=(btz, q_seq_len, self.heads， self.key_size)
        kw = K.reshape(kw, shape=(-1, K.shape(kw)[1], self.heads, self.key_size))  # shape=(btz, k_seq_len, self.heads， self.key_size)
        vw = K.reshape(vw, shape=(-1, K.shape(vw)[1], self.heads, self.head_size))  # shape=(btz, k_seq_len, self.heads， self.head_size)
        # 维度置换 方便计算q*k
        qw = K.permute_dimensions(qw, (0, 2, 1, 3)) # shape=(btz, self.heads，q_seq_len, self.key_size)
        kw = K.permute_dimensions(kw, (0, 2, 1, 3)) # shape=(btz, self.heads，k_seq_len, self.key_size)
        vw = K.permute_dimensions(vw, (0, 2, 1, 3)) # shape=(btz, self.heads，k_seq_len, self.head_size)
        # attention  关于batch_dot的运算方式各位可以看看官方解释，三维好理解，四维也还可以，不过这里的用法会明显不对，计算结束后悔变成shape=(btz, self.heads，q_seq_len, self.heads，k_seq_len)五维
        # a = K.batch_dot(qw, kw, [3, 3]) / self.key_size ** 0.5 # shape=(btz, self.heads，q_seq_len，k_seq_len)

        # 所以个人比较喜欢用tf.enisum 只需要知道矩阵运算后的shape即可
        # 其中b:btz, h:self.heads, j:q_seq_len, k:k_seq_len, d:self.key_size
        a = tf.einsum('bhjd,bhkd->bhjk', qw, kw) / self.key_size ** 0.5 # shape=(btz, self.heads，q_seq_len，k_seq_len)

        # 形状变换，方便mask计算，因为mask的通用形状一般为(btz, seq_len)，a需要对齐
        a = K.permute_dimensions(a, (0, 3, 2, 1)) # shape=(btz, q_seq_len, self.heads, k_seq_len)
        a = sequence_masking(a, v_mask, 1, -1)
        a = K.permute_dimensions(a, (0, 3, 2, 1)) # shape=(btz, self.heads，q_seq_len，k_seq_len)
        if a_mask is not None:
            a = a - (1 - a_mask) * 1e12

        a = K.softmax(a) # shape=(btz, self.heads，q_seq_len，k_seq_len)

        # a的shape=(btz, self.heads，q_seq_len，k_seq_len);
        # vw的shape=(btz, k_seq_len, self.heads， self.head_size)
        # o = K.batch_dot(a, vw, [3, 2]) # shape=(btz, self.heads，q_seq_len，self.head_size)
        o = tf.einsum('bhjk,bkhd->bhjd', a, vw) # shape=(btz, self.heads，q_seq_len，self.head_size)
        o = K.permute_dimensions(o, (0, 2, 1, 3)) # shape=(btz, q_seq_len, self.heads,self.head_size)
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim)) # shape=(btz, q_seq_len, self.out_dim) 因为self.out_dim=self.heads,self.head_size
        if self.key_size != self.head_size:
            o = Dense(
                units=self.head_size,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
        o = sequence_masking(o, q_mask, 0)
        return o # shape=(btz, q_seq_len, self.out_dim)
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)
'''
总结下multiattention的coding思路，可以用6步来解释，先不管mask
input：q,k,v。 注意：自注意力和非自注意力的情况是不一样的
1、全连接层Dense：将最后一维映射成需要的维度 
    q:shape=(btz, q_seq_len, self.heads * self.key_size)
    k:shape=(btz, k_seq_len, self.heads * self.key_size)
    v:shape=(btz, k_seq_len, self.heads * self.head_size) 
2、reshape拆成多头
    qw:shape=(btz, q_seq_len, self.heads， self.key_size)
    kw:shape=(btz, k_seq_len, self.heads， self.key_size)
    vw:shape=(btz, k_seq_len, self.heads， self.head_size)
3、permute_dimensions变换维度方便计算
    qw:shape=(btz, self.heads，q_seq_len, self.key_size)
    kw:shape=(btz, self.heads，k_seq_len, self.key_size)
    vw:shape=(btz, self.heads，k_seq_len, self.head_size)
4、tf.einsum求矩阵a,再进行scale-dot和softmax
    a:shape=(btz, self.heads，q_seq_len，k_seq_len)
5、a和vw矩阵运算
    o:shape=(btz, self.heads，q_seq_len，self.head_size)
6、permute_dimensions和reshape转换成输出维度
    o:shape=(btz, q_seq_len, self.out_dim)
    
参考苏神的bert4keras中的实现，用上tf.einsum只用5步就可以实现
input：q,k,v。 注意：自注意力和非自注意力的情况是不一样的
1、全连接层Dense：将最后一维映射成需要的维度 
    q:shape=(btz, q_seq_len, self.heads * self.key_size)
    k:shape=(btz, k_seq_len, self.heads * self.key_size)
    v:shape=(btz, k_seq_len, self.heads * self.head_size) 
2、reshape拆成多头
    qw:shape=(btz, q_seq_len, self.heads， self.key_size)
    kw:shape=(btz, k_seq_len, self.heads， self.key_size)
    vw:shape=(btz, k_seq_len, self.heads， self.head_size)
3、tf.einsum求矩阵a,再进行scale-dot和softmax
    a:shape=(btz, self.heads，q_seq_len，k_seq_len)
4、a和vw矩阵运算
    o:shape=(btz, self.heads，q_seq_len，self.head_size)
5、reshape转换成输出维度
    o:shape=(btz, q_seq_len, self.out_dim)
'''
        # q, k, v = inputs[:3]
        # if a_mask:
        #     a_mask = inputs[3]
        #
        # qw = self.q_dense(q)
        # kw = self.k_dense(k)
        # vw = self.v_dense(v)
        #
        # # reshape成多头形式
        # qw = K.reshape(qw, shape=(-1, K.shape(q)[1], self.heads, self.key_size))
        # kw = K.reshape(kw, shape=(-1, K.shape(k)[1], self.heads, self.key_size))
        # vw = K.reshape(vw, shape=(-1, K.shape(v)[1], self.heads, self.head_size))
        #
        # # 变换维度
        # # qw = K.permute_dimensions(qw, [0, 2, 1, 3])  # (btz, heads, q_len, key_size)
        # # kw = K.permute_dimensions(kw, [0, 2, 1, 3])
        #
        # # 计算a
        # a = tf.einsum('bjhd,bkhd->bhjk', qw, kw) / self.key_size ** 0.5
        # if a_mask is not None:
        #     a = a - (1 - a_mask) * 1e10
        # a = K.softmax(a)  # （btz,heads,q_len, k_len)
        #
        # # 结果输出
        # o = tf.einsum('bhjk, bkhd->bjhd', a, vw)
        # o = K.reshape(o, shape=(-1, K.shape(o)[1], self.out_dim))
        # if self.key_size != self.head_size:
        #     o = Dense(
        #         units=self.head_size,
        #         use_bias=self.use_bias,
        #         kernel_initializer=self.kernel_initializer
        #     )
        #
        # return o


class MyMultiAttention(Layer):
    def __init__(self, output_dim, num_heads, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.kernel_initializer = kernel_initializer
        super(MyMultiAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(self.num_heads, 3, input_shape[2], self.output_dim),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.Wo = self.add_weight(name='Wo',
                                  shape=(self.num_heads * self.output_dim, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        super(MyMultiAttention, self).build(input_shape)

    def call(self, x):
        q = K.dot(x, self.W[0, 0])
        k = K.dot(x, self.W[0, 1])
        v = K.dot(x, self.W[0, 2])

        e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))
        e = e / (self.output_dim * 0.5)
        e = K.softmax(e)

        o = K.batch_dot(e, v)

        for i in range(1, self.num_heads):
            q = K.dot(x, self.W[i, 0])
            k = K.dot(x, self.W[i, 1])
            v = K.dot(x, self.W[i, 2])

            e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))
            e = e / (self.output_dim * 0.5)
            e = K.softmax(e)

            o = K.batch_dot(e, v)
            output = K.concatenate(output, o)
        z = K.dot(output, self.Wo)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


'''一、读取数据'''
cf = {'data_file_path': r'D:\DP\shuju\20180819\cnews.train\cnews.train.txt',
      'MAX_WORDS': 30000,
      'Tx': 200  # 设置输入新闻的长度为200
      }

with open(cf['data_file_path'], 'r', encoding='utf-8') as f:
    lines = f.readlines()

class_dict = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4,
              '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}
'''提取texts、classes'''
texts = []
classes = []
for line in lines:
    cls = line[:2]
    if (cls in class_dict):
        classes.append(class_dict[cls])
        texts.append(line[3:])
print(len(texts))
print(len(classes))

'''二、数据预处理'''
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tqdm

inputTextList = [' '.join([w for w in jieba.cut(text)]) for text in tqdm.tqdm(texts)]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts=inputTextList)
word_index = tokenizer.word_index
print(len(word_index))

tokenizer.num_word = cf['MAX_WORDS']
input_sequences = tokenizer.texts_to_sequences(texts=inputTextList)

texts_lens = []
for line in tqdm.tqdm(input_sequences):
    texts_lens.append(len(line))
texts_lens.sort()
print('text_len_avg:%f' % (sum(texts_lens) / len(texts)))
print('text_len_middle:%f' % (texts_lens[int(len(texts) / 2)]))
print('text_len_min:%f' % (texts_lens[0]))
print('text_len_max:%f' % (texts_lens[len(texts) - 1]))

import numpy as np

input_arr = []
for line in tqdm.tqdm(input_sequences):
    slen = len(line)
    if (slen < cf['Tx']):
        newline = line + [0] * (cf['Tx'] - slen)
        input_arr.append(newline)
    else:
        input_arr.append(line[:cf['Tx']])
input_arr = np.array(input_arr)
print(input_arr.shape)

from keras.utils import to_categorical

labels = to_categorical(classes)
print(labels.shape)

'''三、定义模型 '''
'''词向量读取'''
with open(r'D:\DP\shuju\retrieval_chatbot\data\w2v\w2v_bk.txt', 'r', encoding='utf-8') as f:
    word_vec = {}
    for line in tqdm.tqdm(f):
        line = line.strip().split()
        curr_word = line[0]
        word_vec[curr_word] = np.array(line[1:], dtype=np.float64)

'''设置词嵌入层'''

from keras.layers import *


def pretrained_embedding_layer(word_vec, word_index):
    vocab_len = cf['MAX_WORDS'] + 1
    emb_dim = 300
    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_index.items():
        vec = word_vec.get(word, np.zeros(emb_dim))
        if (index > cf['MAX_WORDS']):
            break
        emb_matrix[index, :] = vec
        # 定义Embedding层，并指定不需要训练该层的权重
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    embedding_layer.build((None,))  # build
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


embedding_layer = pretrained_embedding_layer(word_vec, word_index)

from keras.models import Sequential, Model
from keras.layers import Dense, SimpleRNN, Embedding, Flatten

seq = Input(shape=(cf['Tx'],))
embed = embedding_layer(seq)
# att = MySelfAttention(128)(embed) # （btz，200，128）
embeds = [embed, embed, embed]
att = MyMultiHeadAttention(heads=4, head_size=32)(embeds)
t = Flatten()(att)
t = Dense(256, activation="relu")(t)
out = Dense(10, activation="softmax")(t)
model = Model(seq, out)
model.summary()
'''四、训练模型'''
out = model.compile(optimizer='rmsprop',
                    metrics=['accuracy'],
                    loss='categorical_crossentropy')

permutation = np.random.permutation(input_arr.shape[0])
x = input_arr[permutation]
y = labels[permutation]
x_val = x[:500]
y_val = y[:500]
x_train = x[500:]
y_train = y[500:]

history = model.fit(x_train, y_train,  # 68,200
                    batch_size=12,
                    epochs=5,
                    verbose=1,
                    validation_data=(x_val, y_val))  # 2000,200

import matplotlib.pyplot as plt


# % matplotlib inline

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
