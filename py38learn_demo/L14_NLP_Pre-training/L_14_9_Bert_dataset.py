import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ==========================================
# 辅助工具：自建工业级词表
# ==========================================
class Vocabulary:
    def __init__(self, tokens, min_freq=5, reserved_tokens=None):
        self.reserved_tokens = reserved_tokens if reserved_tokens else []
        counter = Counter(tokens)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.reserved_tokens)}
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.token_to_idx)
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.unk_idx = self.token_to_idx.get('<unk>', 0)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk_idx)

    def __len__(self):
        return len(self.idx_to_token)


# ==========================================
# 核心流水线：BERT 预训练数据集清洗工厂
# ==========================================
class BERTPretrainDataset(Dataset):
    def __init__(self, data_dir, max_len=64, min_freq=5):
        self.max_len = max_len
        self.max_num_mlm_preds = round(max_len * 0.15)
        
        # 【步骤 1】文本规范化与分词
        paragraphs = self._read_wiki(data_dir)
        
        # 建立词字典
        all_tokens = [token for p in paragraphs for s in p for token in s]
        self.vocab = Vocabulary(all_tokens, min_freq=min_freq, 
                                reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>', '<unk>'])
        
        # 【步骤 2】缝合句子对与 NSP 标签构建
        examples = []
        for paragraph in paragraphs:
            examples.extend(self._get_nsp_data_from_paragraph(paragraph, paragraphs))
            
        processed_examples = []
        for tokens, segments, is_next in examples:
            # 【步骤 3】随机动态掩码 (8-1-1 策略)
            mlm_input_tokens, pred_positions_and_labels = self._get_mlm_data_from_tokens(tokens)
            
            # 【步骤 4】数字化编码 (查字典换 ID)
            token_ids = [self.vocab[token] for token in mlm_input_tokens]
            pred_positions = [pos for pos, _ in pred_positions_and_labels]
            mlm_pred_label_ids = [self.vocab[token] for _, token in pred_positions_and_labels]
            
            processed_examples.append((token_ids, pred_positions, mlm_pred_label_ids, segments, is_next))
            
        # 【步骤 5】截断填充与矩阵对齐
        self.all_token_ids, self.all_segments, self.valid_lens, \
        self.all_pred_positions, self.all_mlm_weights, \
        self.all_mlm_labels, self.nsp_labels = self._pad_bert_inputs(processed_examples)

    def _read_wiki(self, data_dir):
        file_name = os.path.join(data_dir, 'wiki.train.tokens')
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"❌ 路径错误！在 {data_dir} 下未找到 wiki.train.tokens，请确认路径！")
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        paragraphs = []
        for line in lines:
            parts = line.strip().lower().split(' . ')
            if len(parts) >= 2:
                paragraphs.append([sentence.split() for sentence in parts])
        random.shuffle(paragraphs)
        return paragraphs

    def _get_nsp_data_from_paragraph(self, paragraph, paragraphs):
        nsp_data = []
        for i in range(len(paragraph) - 1):
            sentence = paragraph[i]
            if random.random() < 0.5:
                next_sentence = paragraph[i + 1]
                is_next = True
            else:
                next_sentence = random.choice(random.choice(paragraphs))
                is_next = False
            
            if len(sentence) + len(next_sentence) + 3 > self.max_len:
                continue
                
            tokens = ['<cls>'] + sentence + ['<sep>'] + next_sentence + ['<sep>']
            segments = [0] * (len(sentence) + 2) + [1] * (len(next_sentence) + 1)
            nsp_data.append((tokens, segments, is_next))
        return nsp_data

    def _get_mlm_data_from_tokens(self, tokens):
        candidate_pred_positions = []
        for i, token in enumerate(tokens):
            if token in ['<cls>', '<sep>']:
                continue
            candidate_pred_positions.append(i)
            
        num_mlm_preds = max(1, round(len(tokens) * 0.15))
        random.shuffle(candidate_pred_positions)
        
        mlm_input_tokens = list(tokens)
        pred_positions_and_labels = []
        
        for pos in candidate_pred_positions:
            if len(pred_positions_and_labels) >= num_mlm_preds:
                break
            r = random.random()
            if r < 0.8:
                masked_token = '<mask>'
            elif r < 0.9:
                masked_token = tokens[pos]
            else:
                masked_token = random.choice(list(self.vocab.token_to_idx.keys()))
                
            mlm_input_tokens[pos] = masked_token
            pred_positions_and_labels.append((pos, tokens[pos]))
            
        pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
        return mlm_input_tokens, pred_positions_and_labels

    def _pad_bert_inputs(self, examples):
        all_token_ids, all_segments, valid_lens = [], [], []
        all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
        nsp_labels = []
        
        for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
            num_tokens = len(token_ids)
            num_preds = len(pred_positions)
            
            all_token_ids.append(torch.tensor(token_ids + [self.vocab['<pad>']] * (self.max_len - num_tokens), dtype=torch.long))
            all_segments.append(torch.tensor(segments + [0] * (self.max_len - len(segments)), dtype=torch.long))
            valid_lens.append(torch.tensor(num_tokens, dtype=torch.float32))
            
            all_pred_positions.append(torch.tensor(pred_positions + [0] * (self.max_num_mlm_preds - num_preds), dtype=torch.long))
            all_mlm_weights.append(torch.tensor([1.0] * num_preds + [0.0] * (self.max_num_mlm_preds - num_preds), dtype=torch.float32))
            all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (self.max_num_mlm_preds - num_preds), dtype=torch.long))
            nsp_labels.append(torch.tensor(1 if is_next else 0, dtype=torch.long))
            
        return (all_token_ids, all_segments, valid_lens, all_pred_positions, 
                all_mlm_weights, all_mlm_labels, nsp_labels)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx],
                self.all_pred_positions[idx], self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


# ==========================================
# 🎯 3. 点火引擎：MAIN 入口函数
# ==========================================
if __name__ == '__main__':
    print("🎬 正在启动工业级 BERT 数据清洗流水线...")

    # 1. 指定你的本地维基百科数据绝对路径（刚刚我们已经配好了）
    DATA_DIR = '/home/zjianglinux/data/wikitext-2'
    BATCH_SIZE = 512
    MAX_LEN = 64  # 每个样本最大长度阈值
    
    # 2. 实例化数据集工厂（自动在内部执行 步骤1 ~ 步骤5）
    print("⏳ 正在读取本地数据并进行 5 大核心清洗步骤（这可能需要几秒钟）...")
    dataset = BERTPretrainDataset(DATA_DIR, max_len=MAX_LEN)
    print(f"✅ 数据清洗完毕！成功构建专属词表，字典大小 (Vocab Size): {len(dataset.vocab)}")
    
    # 3. 用 PyTorch 的 DataLoader 将清洗好的数据打包成小批次 (Batch)
    train_iter = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # 4. 模拟模型拉取数据，打印第一个批次的张量形状
    print("\n🚀 正在尝试从迭代器中拉取第一个批次 (Batch) 的矩阵数据...")
    for batch in train_iter:
        # 解包 5 大清洗步骤吐给神经网络的 7 个核心张量
        tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y = batch
        
        print("="*50)
        print("🎉 恭喜！数据清洗及批量打包完全成功！形状校验通过：")
        print(f"1. 输入Token矩阵 (tokens_X) 形状:        {list(tokens_X.shape)}    -> [批量大小, 句子最大长度]")
        print(f"2. 句子边界矩阵 (segments_X) 形状:      {list(segments_X.shape)}    -> [区分句A(0)和句B(1)]")
        print(f"3. 有效长度向量 (valid_lens_x) 形状:     {list(valid_lens_x.shape)}    -> [不包含 pad 的真实词数]")
        print(f"4. 考试位置矩阵 (pred_positions_X) 形状: {list(pred_positions_X.shape)}    -> [每句抽考 15% 词元的坐标]")
        print(f"5. 考试损失权重 (mlm_weights_X) 形状:    {list(mlm_weights_X.shape)}    -> [用于过滤 pad 的 0-1 矩阵]")
        print(f"6. 完形填空答案 (mlm_Y) 形状:            {list(mlm_Y.shape)}    -> [抽考词元的正确 Token ID]")
        print(f"7. 下句预测答案 (nsp_y) 形状:            {list(nsp_y.shape)}    -> [是否为真实下句: 1或0]")
        print("="*50)
        
        # 只要成功拉出第一个批次并打印，就说明这套工业清洗流水线完全闭环了！
        break