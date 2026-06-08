import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional



class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos, sin = self.cos[:seq_len, :], self.sin[:seq_len, :]
        x_rotated = torch.zeros_like(x)
        x_rotated[..., ::2] = x1 * cos - x2 * sin
        x_rotated[..., 1::2] = x1 * sin + x2 * cos
        return x_rotated


class AdvancedExpert(nn.Module):
    """采用 SwiGLU 激活函数的顶级专家模块 (Gemma/LLaMA3 同款)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_gate = F.silu(self.w_gate(x))
        return self.drop(self.w_down(swish_gate * self.w_up(x)))

class AdvancedMoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([AdvancedExpert(d_model, d_ff) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        

        self.runtime_stats = {"expert_usage": torch.zeros(num_experts)}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        b, s, d = x.shape
        x_flat = x.view(-1, d)

        raw_logits = self.gate(x_flat)
        if self.training:
            noise = torch.randn_like(raw_logits) * (1.0 / self.num_experts)
            logits = raw_logits + noise
        else:
            logits = raw_logits
            

        gates = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gates, self.k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True) # 重新归一化
        

        P = gates.mean(dim=0)
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        unique_indices, counts = torch.unique(topk_indices, return_counts=True)
        tokens_per_expert[unique_indices] = counts.float()
        F_fraction = tokens_per_expert / (b * s * self.k)
        aux_loss = self.num_experts * torch.sum(P * F_fraction) # 理想状态下 P=F=1/N, 此时 Loss 达到最小
        
        self.runtime_stats["expert_usage"] = tokens_per_expert.detach().cpu()

        # 4. 路由分发
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_idx, expert_pos = torch.where(topk_indices == i)
            if len(token_idx) == 0:
                continue
            expert_out = expert(x_flat[token_idx])
            output[token_idx] += topk_weights[token_idx, expert_pos].unsqueeze(-1) * expert_out
            
        return output.view(b, s, d), aux_loss


class GatedCrossModalBridge(nn.Module):
    """引入信息流调解门控，控制图像和文字的融合深度，防止杂音干扰"""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ln_q = RMSNorm(d_model)
        self.ln_kv = RMSNorm(d_model)
        
        # 门控机制：动态调节交叉注意力信息在骨干网中的流入量
        self.gate_w = nn.Linear(d_model * 2, d_model)

    def forward(self, query_modal: torch.Tensor, key_value_modal: torch.Tensor) -> torch.Tensor:
        q = self.ln_q(query_modal)
        kv = self.ln_kv(key_value_modal)
        
        context_feat, _ = self.mha(query=q, key=kv, value=kv)
        
        # 融合门控计算
        gate = torch.sigmoid(self.gate_w(torch.cat([query_modal, context_feat], dim=-1)))
        return query_modal + gate * context_feat


class QuantumMultimodalEngine(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048):
        super().__init__()
        # 文本处理器 (文本)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionEmbedding(d_model)
        
        self.vision_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            RMSNorm(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
            RMSNorm(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, d_model, kernel_size=4, stride=4, bias=False) # 最终打碎并映射为 d_model 维度
        )
        
        self.bridge = GatedCrossModalBridge(d_model, n_heads)
        self.pre_norm_self = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        
        self.pre_norm_moe = RMSNorm(d_model)
        self.moe = AdvancedMoELayer(d_model=d_model, d_ff=d_ff, num_experts=8, k=2)
        
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, 1) # 回归或分类概率

    def forward(self, text_ids: torch.Tensor, image_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
 
        t_feat = self.token_embed(text_ids)
        t_feat = self.rope(t_feat)
        v_tokens = self.vision_stem(image_tensors)
        v_tokens = v_tokens.flatten(2).permute(0, 2, 1) # 变为 (B, 196, d_model)
        fused_stream = self.bridge(query_modal=t_feat, key_value_modal=v_tokens)

        normED_stream = self.pre_norm_self(fused_stream)
        attn_out, _ = self.self_attn(query=normED_stream, key=normED_stream, value=normED_stream)
        fused_stream = fused_stream + attn_out # 残差跳跃
        
        moe_input = self.pre_norm_moe(fused_stream)
        moe_out, balancing_loss = self.moe(moe_input)
        final_stream = fused_stream + moe_out

        context_summary = self.final_norm(final_stream).mean(dim=1)
        logits = self.output_head(context_summary)
        

        return logits, balancing_loss





# 防梯度炸裂的归一化
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

# LLaMA同款旋转位置编码，让模型对距离更敏感
class RoPE(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x):
        L = x.shape[1]
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos, sin = self.cos[:L, :], self.sin[:L, :]
        out = torch.zeros_like(x)
        out[..., ::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

# Transformer 自注意力，全局上下文
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        nx = self.norm(x)
        out, _ = self.attn(query=nx, key=nx, value=nx)
        return x + out
    

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A))
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B_size, L, D = x.shape

        x_proj = self.x_proj(x)
        delta, B_mat, C_mat = torch.split(x_proj, [D, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))
        A_mat = -torch.exp(self.A_log.float())
        
        # 微分方程离散化
        deltaA = torch.exp(torch.einsum('b l d, n -> b l d n', delta, A_mat.squeeze(0)))
        deltaB_u = torch.einsum('b l d, b l n, b l d -> b l d n', delta, B_mat, x)
        
        # 递归扫描记忆
        h = torch.zeros(B_size, D, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_u[:, t]
            y = torch.einsum('b d n, b n -> b d', h, C_mat[:, t])
            ys.append(y)
            
        return torch.stack(ys, dim=1)

# 动态开辟内存
class NeuralRAM(nn.Module):
    def __init__(self, slots=64, mem_dim=64, d_model=256):
        super().__init__()
        self.slots = slots
        self.mem_dim = mem_dim
        self.r_keys = nn.Linear(d_model, mem_dim)
        self.w_keys = nn.Linear(d_model, mem_dim)
        self.erase = nn.Linear(d_model, mem_dim)
        self.add = nn.Linear(d_model, mem_dim)
        self.gate = nn.Linear(d_model, 1)

    def forward(self, x, mem):
        # 算读写头指针
        r_k, w_k = self.r_keys(x), self.w_keys(x)
        r_score = F.softmax(F.cosine_similarity(r_k.unsqueeze(1), mem, dim=-1), dim=-1)
        w_score = F.softmax(F.cosine_similarity(w_k.unsqueeze(1), mem, dim=-1), dim=-1)
        # 读内存
        read_data = torch.einsum('b s, b s d -> b d', r_score, mem)
        # 写内存
        g = torch.sigmoid(self.gate(x)).unsqueeze(-1)
        e_m = 1 - torch.einsum('b s, b d -> b s d', w_score, torch.sigmoid(self.erase(x))) * g
        a_m = torch.einsum('b s, b d -> b s d', w_score, self.add(x)) * g
        new_mem = mem * e_m + a_m
        
        return read_data, new_mem


class HyperExpert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 算动态矩阵的生成器
        self.w_gen = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model * d_model))
        self.b_gen = nn.Linear(d_model, d_model)
        
    def forward(self, x, context):
        B = x.shape[0]
        # 现场生权重
        w = self.w_gen(context).view(B, self.d_model, self.d_model)
        b = self.b_gen(context).view(B, 1, self.d_model)
        return F.silu(torch.bmm(x, w) + b)


class HierarchicalMoE(nn.Module):
    def __init__(self, d_model, num_experts=3):
        super().__init__()
        self.level1_router = nn.Linear(d_model, 2) # 0:普通流, 1:元学习黑魔法流
        self.level2_router = nn.Linear(d_model, num_experts)
        self.normal_experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.hyper_experts = nn.ModuleList([HyperExpert(d_model) for _ in range(num_experts)])
        
    def forward(self, x, context):
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        ctx_flat = context.repeat(1, L, 1).view(-1, D)
        
        l1_w = F.softmax(self.level1_router(x_flat), dim=-1)
        l2_w = F.softmax(self.level2_router(x_flat), dim=-1)
        
        out = torch.zeros_like(x_flat)
        for e_idx in range(l2_w.shape[-1]):
            out += l1_w[:, 0:1] * l2_w[:, e_idx:e_idx+1] * F.relu(self.normal_experts[e_idx](x_flat))
            out += l1_w[:, 1:2] * l2_w[:, e_idx:e_idx+1] * self.hyper_experts[e_idx](x_flat.unsqueeze(1), ctx_flat).squeeze(1)
            
        return out.view(B, L, D)


class UltimateHybridEngine(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rope = RoPE(d_model)
        
        self.transformer_layer = TransformerBlock(d_model, n_heads)
        self.ssm_layer = SelectiveSSM(d_model)
        
        self.ram = NeuralRAM(slots=64, mem_dim=64, d_model=d_model)
        self.moe = HierarchicalMoE(d_model)
        
        self.mem_proj = nn.Linear(64, d_model)
        self.out_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, L = x.shape
        

        h = self.rope(self.embed(x))
        
        h = self.transformer_layer(h)
        
        h = self.ssm_layer(h)
        
        mem_slots = torch.zeros(B, 64, 64, device=x.device)
        reads = []
        for t in range(L):
            r_data, mem_slots = self.ram(h[:, t, :], mem_slots)
            reads.append(r_data)
        
        fused = h + self.mem_proj(torch.stack(reads, dim=1))
        ctx = fused.mean(dim=1, keepdim=True)
        

        final_features = self.moe(fused, ctx)
        return self.out_head(final_features)

if __name__ == "__main__":

    model = UltimateHybridEngine(vocab_size=5000, d_model=128, n_heads=4)
    
    mock_input = torch.randint(0, 5000, (2, 32))
    logits = model(mock_input)
    