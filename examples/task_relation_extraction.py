#! -*- coding:utf-8 -*-
# 三元组抽取任务，基于“半指针-半标注”结构
# 文章介绍：https://kexue.fm/archives/7161
# 数据集：http://ai.baidu.com/broad/download?dataset=sked
# 最优f1=0.82198
# 换用RoBERTa Large可以达到f1=0.829+
''''''
'''
这个模型是目前苏神写过比较复杂的模型之一，只要把这个模型弄懂了，再看其他的模型就会很简单
概念详解；
1、半指针半标注：我的理解是对subject的预测中，不是常用的直接用NER
而是先预测每个字的是实体的概率，拿到预测出的实体的位置id，通过CLN影响到后续的po，因为通常spo会有关联的，这样更合理
而这个id就是指针，标签的预测就是标注，也就是作者说的半指针半标注

2、多个任务模型应该怎么再同一个代码模型中写出来：
分别定义好各自任务的模型后，再写一个总的模型，并且只需要保存这个总的模型即可：
比如：作者分别建好了subject_model、object_model，总模型为train_model，在预测的时候调用各自的模型即可
'''
import json
import codecs
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import LayerNormalization, Loss
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import *
from keras.models import Model
from tqdm import tqdm


maxlen = 128
batch_size = 64
config_path = 'D:/DP/config/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'D:/DP/config\chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'D:/DP/config/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with codecs.open(filename, encoding='utf-8') as f:
        # 原始数据：l是一个集合，里面有3对
        # {
        #       "postag": [{"word": "茶树茶网蝽", "pos": "nz"}, {"word": "，", "pos": "w"}, {"word": "Stephanitis chinensis Drake", "pos": "nz"}, {"word": "，", "pos": "w"}, {"word": "属", "pos": "v"}, {"word": "半翅目", "pos": "nz"}, {"word": "网蝽科冠网椿属", "pos": "nz"}, {"word": "的", "pos": "u"}, {"word": "一种", "pos": "m"}, {"word": "昆虫", "pos": "n"}],
        #       "text": "茶树茶网蝽，Stephanitis chinensis Drake，属半翅目网蝽科冠网椿属的一种昆虫",
        #       "spo_list": [{"predicate": "目", "object_type": "目", "subject_type": "生物", "object": "半翅目", "subject": "茶树茶网蝽"}]
        # }
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [
                    (spo['subject'], spo['predicate'], spo['object'])
                    for spo in l['spo_list']
                ]
            })
    return D

# 加载数据集
# 得出每一条形式如：{'text': '如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈', 'spo_list': [('喜剧之王', '主演', '周星驰')]}
train_data = load_data('D:/DP/shuju/zhuojunjie-bert4keras-master/bert4keras/examples/datasets/relation/train_data.json')
valid_data = load_data('D:/DP/shuju/zhuojunjie-bert4keras-master/bert4keras/examples/datasets/relation/dev_data.json')

# 构造predicate2id，在data_generator中会使用
predicate2id, id2predicate = {}, {}
with codecs.open('D:/DP/shuju/zhuojunjie-bert4keras-master/bert4keras/examples/datasets/relation/all_50_schemas',encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 检索词在text(词和text均已转成id)中的位置,并返回开头位置的标识符(位置id)
def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

# {'text': '如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈', 'spo_list': [('喜剧之王', '主演', '周星驰')]}
# data_generator根据建模时的input来制作，模型中的subject_labels，subject_ids，object_labels均需在这里制作输入，具体各自的shape如下：
# subject_labels:(?, ?, 2)  subject_ids:(?, 2)   object_labels:(?, ?, 49, 2) 在没进入batch前不用管第一维
class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:  # 随机打乱
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for i in idxs:
            d = self.data[i]  # 取出一条数据
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=maxlen)  # 常规准备预测数据做法
            # 整理三元组 batch_subject_labels, batch_subject_ids, batch_object_labels 转成Input需要的格式 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1] # [0]表示subject的encode只取token，segment不加入，同时[1:-1]去头去尾，只取s主体id
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1] # 同上述s
                s_idx = search(s, token_ids) # 检索s在text的开头位置
                o_idx = search(o, token_ids) # 检索o在text的开头位置
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1) # s开头和结尾位置id
                    o = (o_idx, o_idx + len(o) - 1, p) # o开头和结尾位置id，和p的id
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签 没入btz前 shape=(seq_len, 2)
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1 # s[0]:subject开始位置的id， 0：表示开头的那一行
                    subject_labels[s[1], 1] = 1 # s[1]:subject结束位置的id， 1：表示结尾的那一行
                # 随机选一个subject
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(
                    end[end >= start])  # end >= start 表示用[False,True]来看能不能选，但是这样随机会选到后面的数，导致不是一个真的subject  为什么要这么麻烦呢
                subject_ids = (start, end)
                # 对应的object标签  没入btz前  shape=(seq_len, len(predicate2id), 2)
                object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
                for o in spoes.get(subject_ids, []):  # {(2,4):[(3,5,8)]}
                    object_labels[o[0], o[2], 0] = 1 # o[0]:object开头的位置id， o[2]：对应predict的id， 0：表示开头的那一行
                    object_labels[o[1], o[2], 1] = 1 # o[1]:object结尾的位置id， o[2]：对应predict的id， 1：表示结尾的那一行
                # 构建batch 每一个shape都加一个btz维度
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(batch_subject_labels, padding=np.zeros(2))
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(batch_object_labels,
                                                           padding=np.zeros((len(predicate2id), 2)))
                    yield [
                              batch_token_ids, batch_segment_ids,
                              batch_subject_labels, batch_subject_ids, batch_object_labels
                          ], None # []表示input，在[]外面的表示的是预测的结果，模型自动的做loss。
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


# 取出subject对应的位置的字向量。主要的函数是batch_gather，可以搜索下其用法。
def extrac_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    subject_ids = K.cast(subject_ids, 'int32') # Tensor("lambda_2/Cast:0", shape=(?, 2), dtype=int32)
    start = batch_gather(output, subject_ids[:, :1]) # 取出start的向量 subject_ids表示每个btz中只有一个数字 Tensor("lambda_2/Gather/Reshape_3:0", shape=(?, 1, 768), dtype=float32)
    end = batch_gather(output, subject_ids[:, 1:]) # Tensor("lambda_2/Gather_1/Reshape_3:0", shape=(?, 1, 768), dtype=float32)
    subject = K.concatenate([start, end], 2) # Tensor("lambda_2/concat:0", shape=(?, 1, 1536), dtype=float32)
    return subject[:, 0]


# 补充输入  btz是batchsize的意思
'''
注意:Input定义的参数,比如下面的subject_ids、subject_labels等均需在data_generator中做好输入进来，具体可以看data_generator
（这个问题困扰了我很久，一直以为这个Input是随机初始化的，但其实不是。当你明白了这个框架之后就会如鱼得水）
'''
# Input用法：如果输入进来的第一个位置是None的话，会在位置0再加一维None：输入进来的None表示的是seq_len，，再加一维的None是btz
subject_labels = Input(shape=(None, 2), name='Subject-Labels') # (btz, seq_len, 2) 表示的是每个btz中seq_len长的句子每个字都对应着2个特征，分别表示是subject开头或结尾的概率,从data_generator中可看出其实跟subject_ids是一样的，只是因为这是要预测的，表示形式不同。个人认为在NLU中，这个可以当作slot的实体标签
subject_ids = Input(shape=(2, ), name='Subject-Ids') # (btz, 2) subject_ids表示每个btz中对应是subject开头或者结尾的概率
object_labels = Input(shape=(None, len(predicate2id), 2), name='Object-Labels') # (btz, seq_len, 49, 2):(btz,字数,对应的predicate的个数，首尾)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

# 3.1 先预测subject  简单纯正的bert分类预测任务，对字级别上的预测只需要直接接一个Dense即可
output = Dense(units=2,  # shape = (?, ?, 2) (btz,字数，2)  这里的2表示S首和尾
               activation='sigmoid',
               kernel_initializer=bert.initializer)(bert.model.output) # bert.model.output的输出(btz,seq_len,768)

subject_preds = Lambda(lambda x: x**2)(output) # 把概率平方一下,缓解不平衡问题
# subject_model对应的model
# bert.model.inputs是Input-Token和Input-Segment：0 = {Tensor} Tensor("Input-Token:0", shape=(?, ?), dtype=float32)， 1 = {Tensor} Tensor("Input-Segment:0", shape=(?, ?), dtype=float32)
subject_model = Model(bert.model.inputs, subject_preds)


# 3.2 传入subject_ids，通过指针来影响object的预测
# 通过Conditional Layer Normalization将subject融入到object的预测中
output = bert.model.layers[-2].get_output_at(-1) # (?, ?, 768) bert中最后LN不要做，取出来自己用CLN做 获取某一个网络层的输出；get_output_at：keras函数专用共享编码层
subject = Lambda(extrac_subject)([output, subject_ids]) # shape=(?, 1536) 根据subject_ids从output中取出subject首尾的向量表征
output = LayerNormalization(conditional=True)([output, subject])  # 带上subject的预测 本质是用subject(?, 1536)通过Dense变成（？，1,768）去影响归一化中的beta和gamma，就是影响缩放偏移的那个就可以了
# 输入s_ids 加上对应的p 去预测出o  那这里其实跟p的具体值没啥关系
output = Dense(units=len(predicate2id) * 2, # dense全连接层输出概率
               activation='sigmoid',
               kernel_initializer=bert.initializer)(output)
output = Lambda(lambda x: x**4)(output)
object_preds = Reshape((-1, len(predicate2id), 2))(output) # (?, ?, 49, 2) -1表示不知道第一维的数字的多少，根据后面确定好了之后自动计算出来
# object_model对应的model
object_model = Model(bert.model.inputs + [subject_ids], object_preds)


# 定义loss，把subject和object的预测loss相加即可
class TotalLoss(Loss):
    '''
    subject_loss和object_loss之和，都是二分类交叉熵
    subject_labels是(btz, seq_len, 2)，object_labels是(btz, seq_len, 49, 2)，都是二分类，故只需要用binary_crossentropy即可
    '''
    def compute_loss(self, inputs, mask=None):
        subject_labels, object_labels = inputs[:2]
        subject_preds, object_preds, _ = inputs[2:]
        if mask[4] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[4], K.floatx())
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds) # (btz, seq_len, 2)在最后一维上计算loss
        subject_loss = K.mean(subject_loss, 2) # (btz, seq_len)就像是(btz, seq_len, 1)  每个字的平均loss
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)

        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        return subject_loss + object_loss
# 把loss作为输出训练模型
subject_preds, object_preds = TotalLoss([2, 3])([
    subject_labels, object_labels,subject_preds, object_preds, bert.model.output
])

# 3.3 总训练模型
train_model = Model(bert.model.inputs + [subject_labels, subject_ids, object_labels],
                    [subject_preds, object_preds])


# # mask这一步在干嘛？
# mask = bert.model.get_layer('Sequence-Mask').output_mask # bert中的Mask层  Tensor("Sequence-Mask/Cast:0", shape=(?, ?), dtype=float32)
# # 4.1 算subject的loss
# subject_loss = K.binary_crossentropy(subject_labels, subject_preds) # Tensor("truediv:0", shape=(), dtype=float32)
# subject_loss = K.mean(subject_loss, 2) # 需变成和mask一样的shape
# subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
# # 4.2 算object的loss
# object_loss = K.binary_crossentropy(object_labels, object_preds) # Tensor("Neg_1:0", shape=(?, ?, 49, 2), dtype=float32)
# object_loss = K.sum(K.mean(object_loss, 3), 2) # Tensor("Sum_2:0", shape=(?, ?), dtype=float32)
# object_loss = K.sum(object_loss * mask) / K.sum(mask) # Tensor("truediv_1:0", shape=(), dtype=float32)

# train_model.add_loss(subject_loss + object_loss)
# 4.3 加优化器
AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learning_rate=1e-5)
train_model.compile(optimizer=optimizer)


def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)  # 为预测准备数据
    # 抽取subject
    subject_preds = subject_model.predict([[token_ids], [segment_ids]])  # （btz, seq, 2） btz=1
    # 第一个0表示的是预测的时候btz只要一个，故取0； 最后的[0]表示的是嵌套了一层，取出来
    start = np.where(subject_preds[0, :, 0] > 0.6)[0]  # np.where作用：输出得出元素的坐标，位置id；[0]表示取第一个btz
    end = np.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))  # 得到subjects
    if subjects:
        spoes = []
        token_ids = np.repeat([token_ids], len(subjects),
                              0)  # 因为subjects中有多个，为了配合含有多个的subjects ，故在y轴，也就是行重复多次；也就object_model中的btz概念
        segment_ids = np.repeat([segment_ids], len(subjects), 0)
        subjects = np.array(subjects)  # 从[(),()]转成[[],[]]矩阵模式
        # 传入subject，抽取object和predicate
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.6)
            end = np.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append((subject, predicate1, (_start, _end)))
                        break
        return [
            (
                tokenizer.decode(token_ids[0, s[0]:s[1] + 1], tokens[s[0]:s[1] + 1]),
                id2predicate[p],
                tokenizer.decode(token_ids[0, o[0]:o[1] + 1], tokens[o[0]:o[1] + 1])
            ) for s, p, o in spoes
        ]
    else:
        return []



class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox



def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = codecs.open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' %
                             (f1, precision, recall))
        s = json.dumps(
            {
                'text': d['text'],
                'spo_list': list(T),
                'spo_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            },
            ensure_ascii=False,
            indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        optimizer.apply_ema_weights()
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            train_model.save_weights('best_model_ext.weights')
        optimizer.reset_old_weights()
        print('f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
              (f1, precision, recall, self.best_val_f1))


if __name__ == '__main__':

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()


    train_model.fit_generator(train_generator.forfit(),
                             steps_per_epoch=len(train_generator),
                             epochs=1,
                             callbacks=[evaluator])

else:
    train_model.load_weights('best_model_ext.weights')
