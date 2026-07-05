# 预训练BERT
#

import torch
from torch import nn
from d2l import torch as d2l


# 核心版块逐段拆解

# 代码一共可以分为 4 个核心部分：

#1. 初始化：加载数据与配置模型参数
batch_size, max_len = 512, 64

# ---- 原版 d2l 下载方式（S3 链接已失效，故注释） ----
# train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
# net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
#                     ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
#                     num_layers=2, dropout=0.2, key_size=128, query_size=128,
#                     value_size=128, hid_in_features=128, mlm_in_features=128,
#                     nsp_in_features=128)

# ---- 修复版：跳过网络下载，直接用本地 wikitext-2/wiki.train.txt ----
import random, os
wiki_dir = os.path.join(os.path.dirname(__file__), 'wikitext-2')
tokens_file = os.path.join(wiki_dir, 'wiki.train.tokens')
if not os.path.exists(tokens_file):
    os.symlink(os.path.join(wiki_dir, 'wiki.train.txt'), tokens_file)
paragraphs = d2l._read_wiki(wiki_dir)
train_set = d2l._WikiTextDataset(paragraphs, max_len)
train_iter = torch.utils.data.DataLoader(
    train_set, batch_size, shuffle=True,
    num_workers=d2l.get_dataloader_workers())
vocab = train_set.vocab

net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_blks=2, dropout=0.2, max_len=max_len)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()



# 2. 评分标准：联合损失函数的计算 (_get_batch_loss_bert)

#由于 BERT 既要考完形填空(MLM),又要考下一句预测(NSP),这个函数负责计算这两科的“综合扣分(Loss)

def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下一句子预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

# 3. 闭卷考试：预训练主循环 (train_bert)

def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')


train_bert(train_iter, net, loss, len(vocab), devices, 50)


# 4、一词多义能力测试，检验预训练成果。输入句子，让训练好的 BERT 吐出它对每个词的理解（特征向量）
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X


tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# 词元：'<cls>','a','crane','is','flying','<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]


tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# 词元：'<cls>','a','crane','driver','came','<sep>','he','just',
# 'left','<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]





