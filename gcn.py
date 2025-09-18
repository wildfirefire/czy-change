import torch
import numpy as np
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tree import Tree, head_to_tree, tree_to_adj


class GCNClassifier(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim  # 输入特征维度（与GCN输出维度一致）
        self.args = args
        self.gcn_model = GCNAbsaModel(args, emb_matrix=emb_matrix)  # 核心GCN模型
        self.classifier = nn.Linear(in_dim, args.num_class)  # 情感分类器（3类：积极/消极/中性）
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_dim, args.num_class)
        )

    def forward(self, inputs):
        # 调用核心模型获取特征，返回输出特征及中间通道特征
        outputs, h_sy, h_se, h_csy, h_cse = self.gcn_model(inputs)
        logits = self.classifier(outputs)  # 映射为类别概率
        # logits = self.mlp(outputs)
        return logits, outputs, h_sy, h_se, h_csy, h_cse

    """
       改动要点：
       - 分类头从单层 Linear 升级为 Dropout + LayerNorm + MLP，两层非线性更稳。
       - 支持传入 class_weights（由外部计算），用于缓解类别不平衡（损失处使用）。
       """

    # def __init__(self, args, emb_matrix=None):
    #     super().__init__()
    #     in_dim = args.hidden_dim
    #     self.args = args
    #     self.gcn_model = GCNAbsaModel(args, emb_matrix=emb_matrix)
    #     self.dropout = nn.Dropout(args.cls_dropout if hasattr(args, 'cls_dropout') else 0.2)
    #     self.norm = nn.LayerNorm(in_dim)
    #     self.mlp = nn.Sequential(
    #         nn.Linear(in_dim, in_dim),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(args.cls_dropout if hasattr(args, 'cls_dropout') else 0.2),
    #         nn.Linear(in_dim, args.num_class)
    #     )
    #
    # def forward(self, inputs):
    #     outputs, h_sy, h_se, h_csy, h_cse = self.gcn_model(inputs)
    #     x = self.norm(outputs)
    #     x = self.dropout(x)
    #     logits = self.mlp(x)
    #     return logits, outputs, h_sy, h_se, h_csy, h_cse


class GCNAbsaModel(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        emb_matrix = torch.Tensor(emb_matrix)
        self.emb_matrix = emb_matrix

        self.in_drop = nn.Dropout(args.input_dropout)
        # create embedding layers
        self.emb = nn.Embedding(args.token_vocab_size, args.emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix.cuda(), requires_grad=False)

        self.pos_emb = nn.Embedding(args.pos_vocab_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None  # POS emb
        self.post_emb = nn.Embedding(args.post_vocab_size, args.post_dim, padding_idx=0) if args.post_dim > 0 else None  # position emb

        # rnn layer
        self.in_dim = args.emb_dim + args.post_dim + args.pos_dim
        self.rnn = nn.LSTM(self.in_dim, args.rnn_hidden, 1, batch_first=True, bidirectional=True)

        # attention adj layer
        self.attn = MultiHeadAttention(args.head_num, args.rnn_hidden * 2)
        # self.attn = MultiHeadAttention(args.head_num, self.in_dim)
        self.h_weight = nn.Parameter(torch.FloatTensor(2).normal_(0.5, 0.5))

        # gcn layer
        self.gcn3 = GCN(args, args.hidden_dim, args.num_layers)
        self.gcn2 = GCN(args, args.hidden_dim, args.num_layers)
        self.gcn_common = GCN(args, args.hidden_dim, args.num_layers)

        self.linear1 = nn.Linear(3 * args.hidden_dim, args.hidden_dim)
        self.linear2 = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.linear_2 = nn.Linear(2 * args.hidden_dim, args.hidden_dim)
        # self.gcn1 = GAT(args, args.hidden_dim, args.num_layers, self.in_dim)
        # 修改后（正确）：
        # 输入维度应为双向LSTM的输出维度：2 * args.rnn_hidden
        self.gcn1 = GAT(args, args.hidden_dim, args.num_layers, 2 * args.rnn_hidden)
        # 豆包改师姐
        self.fusion_doubao = MultiInfoFusion1(args.hidden_dim)
        # 深度交叉融合
        self.fusion4 = EnhancedFusion(args.hidden_dim).to(args.device)
        #自适应注意力融合
        self.fusion_adaptive = AdaptiveFusion(args.hidden_dim)
        #残差融合
        self.fusion_res = ResidualFusion(args.hidden_dim)
        #多层次融合
        self.fusion_layer=HierarchicalFusion(args.hidden_dim)
        # MLP Layer
        self.linear = nn.Linear(3 * args.hidden_dim, args.hidden_dim)

        # 样本级门控
        self.fusion5 = DynamicGatedFusion(args.hidden_dim).to(args.device)
        #门控
        self.fusion1 = GatedFusion(args.hidden_dim).to(args.device)

        self.pool = MultiHeadAttentionPooling(hidden_dim=args.hidden_dim,
                                              num_heads=3,  # 新增参数
                                              dropout=0.1)  # 新增参数
        self.pool1 = CrossAttention(args.hidden_dim, dropout=0.1)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        # 初始化LSTM隐藏状态（全零）
        h0, c0 = rnn_zero_state(batch_size, self.args.rnn_hidden, 1, True)
        # 处理变长序列：打包padding后的序列，提升LSTM效率
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens.cpu(), batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))  # LSTM前向传播
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)  # 解包为padding序列
        return rnn_outputs  # [batch_size, seq_len, 2*rnn_hidden]

    def create_embs(self, tok, pos, post):
        word_embs = self.emb(tok)  # 词嵌入 [batch_size, seq_len, emb_dim]
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs.append(self.pos_emb(pos))  # POS嵌入 [batch_size, seq_len, pos_dim]
        if self.args.post_dim > 0:
            embs.append(self.post_emb(post))  # 位置嵌入 [batch_size, seq_len, post_dim]
        embs = torch.cat(embs, dim=2)  # 拼接为 [batch_size, seq_len, in_dim]
        embs = self.in_drop(embs)  # 输入dropout
        return embs

    def inputs_to_att_adj(self, input, score_mask):
        # 多头注意力计算注意力权重 [batch_size, head_num, seq_len, seq_len]
        attn_tensor = self.attn(input, input, score_mask)
        attn_tensor = torch.sum(attn_tensor, dim=1)  # 多头注意力结果求和 [batch_size, seq_len, seq_len]
        # 筛选top_k个重要连接，生成稀疏邻接矩阵
        attn_tensor = select(attn_tensor, self.args.top_k) * attn_tensor
        return attn_tensor  # 语义邻接矩阵

    def forward(self, inputs):
        # 解析输入：tokens、方面词掩码、POS标签、句法头、位置标签、依存关系、掩码等
        tok, asp, pos, head, post, dep, mask, l, adj ,graph,fro,to= inputs
        embs = self.create_embs(tok, pos, post)  # 生成嵌入特征
        orimask=mask
        # LSTM编码：得到上下文感知的词表示 [batch_size, seq_len, 2*rnn_hidden]
        rnn_hidden = self.encode_with_rnn(embs, l, tok.size(0))
        base_hidden = rnn_hidden

        # 生成注意力掩码（屏蔽padding位置）
        score_mask = torch.matmul(base_hidden, base_hidden.transpose(-2, -1)) == 0  # 零向量位置为True（需屏蔽）
        score_mask = score_mask.unsqueeze(1).repeat(1, self.args.head_num, 1, 1).cuda()

        # 生成语义邻接矩阵（基于注意力）
        att_adj = self.inputs_to_att_adj(base_hidden, score_mask)

        # 多通道GCN特征提取
        # h_sy = self.gcn1(adj, rnn_hidden, score_mask, 'syntax')  # 句法通道（输入句法邻接矩阵）
        # h_se = self.gcn1(att_adj, rnn_hidden, score_mask, 'semantic')  # 语义通道（输入注意力邻接矩阵）
        # h_csy = self.gcn1(adj, rnn_hidden, score_mask, 'syntax')  # 公共通道-句法
        # h_cse = self.gcn1(att_adj, rnn_hidden, score_mask, 'semantic')  # 公共通道-语义
        # h_com = (self.h_weight[0] * h_csy + self.h_weight[1] * h_cse) / 2  # 公共通道特征加权融合

        h_sy = self.gcn1(adj, base_hidden, score_mask, 'syntax')
        # h_sy = self.gcn1(adj, rnn_hidden, score_mask, 'syntax')  # 句法通道（输入句法邻接矩阵）
        h_se = self.gcn1(att_adj, base_hidden, score_mask, 'semantic')  # 语义通道（输入注意力邻接矩阵）
        h_sentic = self.gcn1(graph, base_hidden, score_mask, 'sentic')  # 公共通道-句法

        # 方面词特征池化（仅保留方面词相关特征）
        asp_wn = mask.sum(dim=1).unsqueeze(-1)  # 方面词数量（用于平均）
        mask = mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)  # 扩展掩码维度
        # h_sy_mean = (h_sy * mask).sum(dim=1) / asp_wn  # 句法通道特征均值
        # h_se_mean = (h_se * mask).sum(dim=1) / asp_wn  # 语义通道特征均值
        # h_com_mean = (h_com * mask).sum(dim=1) / asp_wn  # 公共通道特征均值
        # 句法
        h_sy_mean = (h_sy * mask).sum(dim=1) / asp_wn
        # 语义
        h_se_mean = (h_se * mask).sum(dim=1) / asp_wn
        # 情感
        h_sen_mean = (h_sentic * mask).sum(dim=1) / asp_wn
        # h_sen_mean = (h_com * mask).sum(dim=1) / asp_wn
        # print("h_sy.shape:", h_sy.shape)  # 期望 [B,L,D]
        # print("mask shape:", mask.shape)  # 期望 [B,L]
        # mask = mask[:, :, 0]  # [B, L]
        # h_sy_mean = self.pool1(h_sy, mask)
        # h_se_mean = self.pool1(h_se, mask)
        # h_sen_mean = self.pool1(h_sentic, mask)
        # 融合3种通道特征，通过MLP输出
        # outputs = torch.cat((h_sy_mean, h_se_mean, h_sen_mean), dim=-1)
        # outputs = F.relu(self.linear(outputs))
        outputs = self.fusion4(h_sy_mean, h_se_mean, h_sen_mean)  # 深度交叉融合
        # outputs = self.fusion_doubao(h_sy_mean, h_se_mean, h_sen_mean)  # 豆包改融合
        #自适应注意力融合
        # outputs = self.fusion_adaptive(h_sy_mean, h_se_mean, h_sen_mean)
        #残差融合
        # outputs = self.fusion_res(h_sy_mean, h_se_mean, h_sen_mean)
        # outputs = self.fusion_res(h_sy_mean, h_se_mean, h_sen_mean) + outputs
        #多层次融合
        # outputs = self.fusion_layer(h_sy_mean, h_se_mean, h_sen_mean)
        # outputs = self.fusion1(h_sy_mean, h_se_mean, h_sen_mean)  # 门控融合
        outputs = F.relu(outputs)
        h_csy=''
        h_cse=''
        return outputs, h_sy, h_se, h_csy, h_cse


## 修改后的 MHAP 模块


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, h, mask):
        B, L, D = h.size()
        x = self.proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,d]

        q = self.query.unsqueeze(0).unsqueeze(2)  # [1,H,1,d]
        scores = (x * q).sum(dim=-1) / math.sqrt(self.head_dim)  # [B,H,L]

        # 兼容 mask: [B,L] 或 [B,L,D]
        if mask is not None:
            if mask.dim() == 3:
                mask = mask[:, :, 0]  # 压缩到 [B,L]
            m = (mask == 0).unsqueeze(1)  # [B,1,L]
            scores = scores.masked_fill(m, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn).unsqueeze(-1)  # [B,H,L,1]

        # 正确的加权求和
        pooled = (attn * x).sum(dim=2)     # [B,H,d]
        pooled = pooled.transpose(1, 2).contiguous().view(B, D)  # [B,D]

        return self.out(pooled)




class GAT(nn.Module):
    def __init__(self, args, mem_dim, num_layers, input_dim):
        super(GAT, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim  # 输出特征维度
        self.in_dim = input_dim  # 输入特征维度
        self.head_num = args.head_num
        self.d_k = self.mem_dim // self.head_num  # 每个头的维度

        self.in_drop = nn.Dropout(args.input_dropout)
        self.gcn_drop = nn.Dropout(args.gcn_dropout)
        self.dropblock = DropBlock()

        # 1. 将 BatchNorm 替换为 LayerNorm（针对特征维度归一化）
        self.norms = nn.ModuleList([nn.LayerNorm(mem_dim) for _ in range(num_layers)])

        # 2. 新增输入维度适配层（解决第一层输入维度与输出维度不匹配问题）
        self.input_proj = nn.Linear(input_dim, mem_dim) if input_dim != mem_dim else None

        self.gat_layers = nn.ModuleList()
        for layer in range(self.layers):
            # 输入维度：第一层为mem_dim（已通过input_proj转换），后续层为mem_dim
            layer_in_dim = mem_dim
            # 多头投影层
            self.gat_layers.append(
                nn.ModuleList([nn.Linear(layer_in_dim, self.d_k) for _ in range(self.head_num)])
            )
            # 注意力分数计算层
            self.gat_layers.append(
                nn.Linear(2 * self.d_k, 1)
            )
            # print(f"GAT input_dim: {input_dim}, mem_dim: {mem_dim}")

    def calc_attn_score(self, h_i, h_j, attn_linear):
        cat_feat = torch.cat([h_i.unsqueeze(2).repeat(1, 1, h_j.size(1), 1),
                              h_j.unsqueeze(1).repeat(1, h_i.size(1), 1, 1)], dim=-1)
        attn_score = attn_linear(cat_feat).squeeze(-1)
        attn_score = F.leaky_relu(attn_score, negative_slope=0.2)
        return attn_score

    def gat_layer_forward(self, layer_idx, x, adj, score_mask):
        batch_size, seq_len, _ = x.size()
        head_outputs = []

        head_projs = self.gat_layers[2 * layer_idx]
        attn_linear = self.gat_layers[2 * layer_idx + 1]

        # 3. 第一层输入维度转换（若输入维度≠输出维度）
        if layer_idx == 0 and self.input_proj is not None:
            x = self.input_proj(x)  # 将输入维度转换为mem_dim

        for head in range(self.head_num):
            proj_x = head_projs[head](x)  # (batch, seq_len, d_k)
            attn_score = self.calc_attn_score(proj_x, proj_x, attn_linear)

            if score_mask is not None:
                head_mask = score_mask[:, head, :, :] if score_mask.size(1) == self.head_num else score_mask[:, 0, :, :]
                attn_score = attn_score.masked_fill(head_mask, -1e9)

            if adj is not None:
                attn_score = attn_score.masked_fill(adj == 0, -1e9)

            attn_weight = F.softmax(attn_score, dim=-1)
            attn_weight = self.gcn_drop(attn_weight)
            head_out = torch.bmm(attn_weight, proj_x)  # (batch, seq_len, d_k)
            head_outputs.append(head_out)

        # 多头融合（确保输出维度为mem_dim）
        concat_out = torch.cat(head_outputs, dim=-1)  # (batch, seq_len, mem_dim)

        # 4. 残差连接 + LayerNorm（此时x和concat_out维度均为mem_dim）
        residual_out = x + concat_out  # 残差连接：输入+输出
        norm_out = self.norms[layer_idx](residual_out)  # LayerNorm归一化

        if layer_idx < self.layers - 1:
            norm_out = self.gcn_drop(norm_out)

        return norm_out

    def forward(self, adj, inputs, score_mask, type):

        # 计算当前序列长度
        seq_len = inputs.size(1)
        dynamic_block_size = max(1, min(3, seq_len // 5))
        x = self.dropblock(inputs,dynamic_block_size)
        x = self.in_drop(x)

        for layer_idx in range(self.layers):
            if type == 'semantic' and layer_idx > 0:
                adj = None
            x = self.gat_layer_forward(layer_idx, x, adj, score_mask)

        return x


class DropBlock(nn.Module):
    def __init__(self, keep_prob=0.9):  # 移除固定block_size，仅保留概率参数
        super().__init__()
        self.keep_prob = keep_prob
        self.gamma = None  # 动态计算丢弃概率

    def forward(self, x, block_size):  # 新增block_size参数
        if not self.training:
            return x

        # 对于序列数据（shape: [batch_size, seq_len, hidden_dim]）
        batch_size, seq_len, hidden_dim = x.shape

        # 计算gamma（控制丢弃概率，与block_size相关）
        self.gamma = (1.0 - self.keep_prob) / (block_size ** 2)

        # 生成掩码（伯努利分布）
        mask = torch.bernoulli(torch.full((batch_size, seq_len, 1), self.gamma, device=x.device))
        mask = mask.expand(-1, -1, hidden_dim)  # 扩展到特征维度

        # 将掩码转换为block形式（连续block_size长度的区域被丢弃）
        # 1. 对序列维度进行膨胀，便于后续卷积生成block
        mask = mask.unsqueeze(1)  # [batch, 1, seq_len, hidden]
        # 2. 使用卷积生成连续block（确保block内全为1）
        conv = nn.Conv2d(1, 1, kernel_size=(block_size, 1), stride=1, padding=(block_size // 2, 0), bias=False)
        conv.weight.data.fill_(1.0)
        conv = conv.to(x.device)
        with torch.no_grad():
            mask = conv(mask)  # 卷积后，连续block_size的位置会被标记为1
        mask = mask.squeeze(1)  # [batch, seq_len, hidden]
        mask = (mask > 0).float()  # 二值化：block区域为1，其他为0

        # 应用掩码（丢弃block区域，保留其他区域并rescale）
        x = x * (1 - mask)
        x = x * (mask.numel() / (mask.numel() - mask.sum()))  # rescale保留值
        return x
class GCN(nn.Module):
    def __init__(self, args, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.layers = num_layers  # GCN层数
        self.mem_dim = mem_dim  # 每层输出维度
        self.in_dim = args.rnn_hidden * 2  # 输入维度（LSTM输出的2*rnn_hidden）

        # Dropout层
        self.in_drop = nn.Dropout(args.input_dropout)
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # GCN层参数：线性变换矩阵（W）和注意力层（attn，用于动态更新邻接矩阵）
        self.W = nn.ModuleList()  # 每层的线性变换
        self.attn = nn.ModuleList()  # 从第二层开始的注意力层（用于生成动态邻接矩阵）
        for layer in range(self.layers):
            input_dim = self.in_dim + layer * self.mem_dim  # 第l层输入维度（累加前l-1层输出）
            self.W.append(nn.Linear(input_dim, self.mem_dim))  # 线性变换矩阵
            if layer != 0:  # 第一层无需注意力（用初始邻接矩阵）
                self.attn.append(MultiHeadAttention(args.head_num, input_dim))

    def GCN_layer(self, adj, gcn_inputs, denom, l):
        Ax = adj.bmm(gcn_inputs)  # 邻接矩阵 × 输入特征（聚合邻居信息）
        AxW = self.W[l](Ax)  # 线性变换
        AxW = AxW / denom  # 归一化（除以节点度+1，避免数值过大）
        gAxW = F.relu(AxW) + self.W[l](gcn_inputs)  # 残差连接（自身特征+邻居聚合特征）
        # 除最后一层外，应用dropout防止过拟合
        gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return gcn_inputs

    def forward(self, adj, inputs, score_mask, type):
        denom = adj.sum(2).unsqueeze(2) + 1  # 计算归一化系数（节点度+1）
        out = self.GCN_layer(adj, inputs, denom, 0)  # 第一层GCN（用初始邻接矩阵）

        # 从第二层开始，动态更新邻接矩阵（仅语义通道）
        for i in range(1, self.layers):
            inputs = torch.cat((inputs, out), dim=-1)  # 拼接历史特征（输入+前层输出）

            if type == 'semantic':  # 语义通道：动态生成新邻接矩阵
                adj = self.attn[i - 1](inputs, inputs, score_mask)  # 注意力生成邻接矩阵
                # 处理多头注意力结果（取最大头或求和）
                if self.args.second_layer == 'max':
                    probability = F.softmax(adj.sum(dim=(-2, -1)), dim=0)
                    max_idx = torch.argmax(probability, dim=1)
                    adj = torch.stack([adj[b][max_idx[b]] for b in range(len(max_idx))], dim=0)
                else:
                    adj = torch.sum(adj, dim=1)
                adj = select(adj, self.args.top_k) * adj  # 筛选top_k连接
                denom = adj.sum(2).unsqueeze(2) + 1  # 更新归一化系数

            out = self.GCN_layer(adj, inputs, denom, i)  # 第i层GCN
        return out


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # 拼接后降维
        self.dropout = nn.Dropout(dropout)

    def forward(self, aspect, context, context_mask=None):
        """
        aspect: [B, La, D]  (aspect tokens 表示)
        context: [B, Lc, D] (上下文表示)
        context_mask: [B, Lc] (0=padding,1=有效)
        """
        # --- aspect attends to context ---
        Q_a = self.q_proj(aspect)    # [B,La,D]
        K_c = self.k_proj(context)   # [B,Lc,D]
        V_c = self.v_proj(context)   # [B,Lc,D]

        scores_ac = torch.matmul(Q_a, K_c.transpose(-2,-1)) / (K_c.size(-1) ** 0.5)  # [B,La,Lc]
        if context_mask is not None:
            scores_ac = scores_ac.masked_fill(context_mask.unsqueeze(1)==0, -1e9)
        attn_ac = F.softmax(scores_ac, dim=-1)
        aspect_enhanced = torch.matmul(attn_ac, V_c)  # [B,La,D]

        # --- context attends to aspect ---
        Q_c = self.q_proj(context)
        K_a = self.k_proj(aspect)
        V_a = self.v_proj(aspect)

        scores_ca = torch.matmul(Q_c, K_a.transpose(-2,-1)) / (K_a.size(-1) ** 0.5)  # [B,Lc,La]
        attn_ca = F.softmax(scores_ca, dim=-1)
        context_enhanced = torch.matmul(attn_ca, V_a)  # [B,Lc,D]

        # --- 融合: 取均值池化，再拼接 ---
        aspect_vec = aspect_enhanced.mean(dim=1)       # [B,D]
        context_vec = context_enhanced.mean(dim=1)     # [B,D]
        out = torch.cat([aspect_vec, context_vec], dim=-1)  # [B,2D]

        return self.out_proj(out)  # [B,D]

#深度交叉融合
class EnhancedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 自注意力增强单特征
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=2)
        # 交叉融合（复用原逻辑）
        self.cross_fusion = MultiInfoFusion1(hidden_dim)
    def forward(self, t1, t2, t3):
        # 自增强：每个特征先通过自注意力聚焦关键信息
        t1_enhanced = self.self_attn(t1.unsqueeze(0), t1.unsqueeze(0), t1.unsqueeze(0))[0].squeeze(0)
        t2_enhanced = self.self_attn(t2.unsqueeze(0), t2.unsqueeze(0), t2.unsqueeze(0))[0].squeeze(0)
        t3_enhanced = self.self_attn(t3.unsqueeze(0), t3.unsqueeze(0), t3.unsqueeze(0))[0].squeeze(0)
        # 交叉融合增强后的特征
        return self.cross_fusion(t1_enhanced, t2_enhanced, t3_enhanced)
class MultiInfoFusion1(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate12 = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())
        self.gate13 = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())
        self.gate23 = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())
        self.transform12 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.transform13 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.transform23 = nn.Linear(2 * hidden_dim, hidden_dim)
        # 注意力融合（替代卷积）
        self.attn_fusion = nn.Linear(hidden_dim, 1)  # 对3个特征计算注意力权重

    def forward(self, t1, t2, t3):
        # 局部门控（同原逻辑）
        t12g = self.transform12(torch.cat([t1, t2], -1)) * self.gate12(torch.cat([t1, t2], -1))
        t13g = self.transform13(torch.cat([t1, t3], -1)) * self.gate13(torch.cat([t1, t3], -1))
        t23g = self.transform23(torch.cat([t2, t3], -1)) * self.gate23(torch.cat([t2, t3], -1))

        # 注意力全局融合
        fused_stacked = torch.stack([t12g, t13g, t23g], dim=1)  # (batch, 3, hidden_dim)
        attn_weights = F.softmax(self.attn_fusion(fused_stacked).squeeze(-1), dim=1)  # (batch, 3)
        final_out = (fused_stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)
        return final_out
#自适应注意力融合
class AdaptiveFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)  # 每个通道学一个打分

    def forward(self, f1, f2, f3):
        feats = torch.stack([f1, f2, f3], dim=1)  # [B,3,H]
        scores = self.proj(feats).squeeze(-1)     # [B,3]
        weights = F.softmax(scores, dim=1)        # [B,3]
        fused = (feats * weights.unsqueeze(-1)).sum(dim=1)  # 加权融合
        return fused
#残差融合
class ResidualFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(3*hidden_dim, hidden_dim)

    def forward(self, f1, f2, f3):
        concat = torch.cat([f1, f2, f3], dim=-1)   # [B, 3H]
        fused = F.relu(self.linear(concat))        # 压缩
        return fused + (f1 + f2 + f3) / 3          # 残差增强

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape))  # 改为param
    return h0.cuda(), c0.cuda()


def select(matrix, top_num):
    batch = matrix.size(0)
    len = matrix.size(1)
    matrix = matrix.reshape(batch, -1)
    maxk, _ = torch.topk(matrix, top_num, dim=1)

    for i in range(batch):
        matrix[i] = (matrix[i] >= maxk[i][-1])
    matrix = matrix.reshape(batch, len, len)
    matrix = matrix + matrix.transpose(-2, -1)

    # selfloop
    for i in range(batch):
        matrix[i].fill_diagonal_(1)

    return matrix


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, hidden_dim, dropout=0.1):
        super().__init__()
        assert hidden_dim % head_num == 0  # 确保隐藏维度可被头数整除
        self.d_k = hidden_dim // head_num  # 每个头的维度
        self.head_num = head_num
        self.linears = clones(nn.Linear(hidden_dim, hidden_dim), 2)  # 两个线性层（query和key）
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, score_mask, dropout=None):
        d_k = query.size(-1)
        # 注意力分数 = (query × key^T) / sqrt(d_k)（缩放点积）
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if score_mask is not None:
            scores = scores.masked_fill(score_mask, -1e9)  # 屏蔽无效位置（padding）
        # 计算注意力权重（softmax），并再次屏蔽padding
        b = ~score_mask[:, :, :, 0:1]  # 有效位置掩码（True为有效）
        p_attn = F.softmax(scores, dim=-1) * b.float()
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn

    def forward(self, query, key, score_mask):
        nbatches = query.size(0)
        # 线性变换 + 多头拆分：[batch, seq_len, hidden] → [batch, head_num, seq_len, d_k]
        query, key = [l(x).view(nbatches, -1, self.head_num, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        attn = self.attention(query, key, score_mask, dropout=self.dropout)  # 多头注意力权重
        return attn  # [batch_size, head_num, seq_len, seq_len]

#样本级门控
class DynamicGatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3*hidden_dim, 3),  # 输入所有特征拼接
            nn.Softmax(dim=1)  # 样本级权重归一化
        )
    def forward(self, t1, t2, t3):
        weights = self.gate(torch.cat([t1, t2, t3], dim=-1))  # (batch, 3)
        return t1 * weights[:,0:1] + t2 * weights[:,1:2] + t3 * weights[:,2:3]
class GatedFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.gate = nn.Linear(3 * feat_dim, 3)  # 门控权重计算

    def forward(self, f1, f2, f3):
        concat_feat = torch.cat([f1, f2, f3], dim=-1)
        gate_weights = torch.sigmoid(self.gate(concat_feat))  # (B, 3)
        fused = f1 * gate_weights[:, 0:1] + f2 * gate_weights[:, 1:2] + f3 * gate_weights[:, 2:3]
        return fused

#多层次融合
class HierarchicalFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fuse1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fuse2 = nn.Linear(2*hidden_dim, hidden_dim)

    def forward(self, f1, f2, f3):
        f12 = F.relu(self.fuse1(torch.cat([f1, f2], dim=-1)))  # 融合句法+语义
        f123 = F.relu(self.fuse2(torch.cat([f12, f3], dim=-1))) # 再融合情感
        return f123
