# ! -*- coding: utf-8 -*-
# bert做language model任务，小说生成

'''
实现思路：
一、数据处理，处理成训练数据
    1.1 清洗数据
    1.2 处理成训练数据
二、建模
    2.1 bert
    2.2 loss和optimize
三、Data_generator

四、evaluate测试
    4.1 生成句子

'''
from __future__ import print_function
import glob, re
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model

maxlen =50
batch_size = 2
steps_per_epoch = 1000
epochs = 20

# bert配置
config_path = r'D:\DP\config\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\DP\config\chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = r'D:\DP\config\chinese_L-12_H-768_A-12/vocab.txt'
novels = []
data_path = 'D:/DP/shuju/zhuojunjie-bert4keras-master/bert4keras/examples/datasets/金庸/*.txt'

# 一、数据处理，处理成训练数据
# 1.1 清洗数据
'''
数据格式：
最终novels为[[句子1，句子2，...，句子n],  # 文章1
             [句子1，句子2，...，句子m],  # 文章2
             ...
             [句子1，句子2，...，句子z]]
由多篇文章组成的列表，每个子列表由句子组成
'''
for txt in glob.glob(data_path):
    txt = open(txt, encoding='utf-8').read()
    txt = txt.replace('\r', '').replace('\n', '') # 清洗掉 \r 和 \n，对于文本的清洗可以直接用replace
    txt = txt.replace(u'整理制作，并提供下载', '') # 同上
    # txt = re.sub(u'www.*?com', '', txt) # 这里我在跑的时候会让txt后面的都没了，故注释掉
    txt = txt.replace(u'\u3000', ' ') # 同上
    sents = []
    for t in txt.split('  '):
        for s in re.findall(u'.*?。', t): # u'.*?。'表示找出所有以。为结尾的，也就是代表着句子
            if len(s) <= maxlen - 2: # 只有句子不要超过maxlen - 2才能够进入训练器中，要不然太长会太稀疏，影响预测结果
                sents.append(s)
    novels.append(sents) # 一篇文章中所有的句子 最后是几篇文章的列表

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

# 1.2 处理成训练数据
data = [] # 得到训练数据，由长度不大于(maxlen - 2)的训练句子
pbar = tqdm(desc=u'构建语料中', total=sum(len(n) for n in novels))
# 以每句话为开头  后面接不超过maxlen个字  表示着后面的续写
for novel in novels:
    s = u'' # 目的是：以每篇文章中每句话为开头，得到以其为开头的长度不大于(maxlen - 2)的训练句子（减2是因为生成batch的时候会有CLS和SEP两个占位，需提前预留好）
    for i in range(len(novel)): # 这里的i表示的是每句话都会为开头
        for j in range(len(novel) - i): # len(novel) - i：表示后面取句子的时候，只取i后面的句子
            if len(s) + len(novel[i + j]) > maxlen - 2: # 每句话都作为一个开始，直到超过maxlen后停止
                data.append(s)
                s = u''
                break
            else:
                s += novel[i + j] # 每句话都作为一个开始，直到超过maxlen后停止
        pbar.update(1) # 更新进度条
        if i + j >= len(novel):
            break
    if s:
        data.append(s)

pbar.close()
np.random.shuffle(data) # 打乱数据

# 三、Data_generator 只需要对输入文字tokenizer一下即可
class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, text in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text) # 只需要对输入文字tokenizer一下即可
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                try:
                    yield [batch_token_ids, batch_segment_ids], None
                except StopIteration:
                    return
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs # y_true：(btz, seq_len)  y_pred:(btz, seq_len, 13584)
        if mask[1] is None:
            y_mask = 1.0
        else:
            y_mask = K.cast(mask[1], K.floatx())[:, 1:] # 去掉头部的 CLS 因为预测的时候没用到它
        y_true = y_true[:, 1:]  # 目标token_ids，从第二个字开始预测，所以取第二个字为开头
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位 （btz, seq_len, 13584）
        loss = K.sparse_categorical_crossentropy(y_true, y_pred) # (btz, seq_len) sparse_categorical_crossentropy的用法是不需要自己转成one_hot形式
        loss = K.sum(loss * y_mask) / K.sum(y_mask) # 算每个btz中loss的平均值，mask掉padding部分，也就是把padding的loss去掉
        return loss

# 二、建模
# 2.1 bert
model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='lm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)
# 2.2 loss  字级别的预测，13584表示13584个字中每个字是下一个字的概率
output = CrossEntropy(1)([model.inputs[0], model.outputs[0]]) # model.inputs[0]是batch_token_ids原句字就是正确的答案  model.outputs[0]:shape=(btz, seq_len 13584)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class StoryCompletion(AutoRegressiveDecoder):
    """基于随机采样的故事续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0]
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.zeros_like(token_ids)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids[:-1]], n, topk)  # 基于随机采样 可以跳到AutoRegressiveDecoder查看其原理
        return [text + tokenizer.decode(ids) for ids in results]

# 输入的为AutoRegressiveDecoder需要初始化的参数
story_completion = StoryCompletion(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)

# 4.1 生成句子
def just_show():
    s1 = u'当晚两人在一家小客店中宿歇。张无忌躺在炕上，越想越是担心，走到赵敏窗外，但听她呼吸调匀，正自香梦沉酣。'
    s2 = u'虚竹飞身跃上松树的枝干，只见段延庆的钢杖深深嵌在树枝之中，全凭一股内力粘劲，挂住了下面四人，内力之深厚，实是非同小可。虚竹伸左手抓住钢杖，提将上来。'
    s3 = u'杨过居住在侠客岛，是令狐冲的弟子，武器是金蛇剑。'
    for s in [s1, s2, s3]:
        t = story_completion.generate(s)
        print(u'输入: %s' % s)
        print(u'结果: %s\n' % ('\n'.join(t)))

# 四、evaluate测试
class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        # 用loss作为评价指标
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model_lm.weights')
        # 演示效果
        just_show()


# if __name__ == '__main__':
if None:
    evaluator = Evaluate()
    train_generator = data_generator(data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model_lm.weights')
    just_show()