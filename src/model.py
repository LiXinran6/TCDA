#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from src.Roberta import MultiHeadAttention, InteractionAttention
from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
from itertools import accumulate

import torch.nn.functional as F

# 逐位置前馈网络
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1, d_out=None):
        super(PositionwiseFeedForward, self).__init__()
        if d_out is None: d_out = d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

# 让 “局部词向量（Local Token）” 和 “全局句向量（Global Utterance）” 进行深度信息交换。
class InteractLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, config=None):
        super(InteractLayer, self).__init__()
        head_size = int(d_model / num_heads)
        self.config = config
        self.interactionAttention = InteractionAttention(num_heads, d_model, head_size, head_size, dropout, config=config)

        self.layer_norm_pre = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.layer_norm_post = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, global_x, mask, sentence_length,):
        x = self.layer_norm_pre(self.interactionAttention(x, global_x, mask,)[0] + x)
        x = self.layer_norm_post(self.ffn(x) + x)
        x = self.dropout(x)
        return x

class DAG_RGCN(nn.Module):
    def __init__(self, hidden_dim):
        super(DAG_RGCN, self).__init__()
        self.hidden_dim = hidden_dim
        # RGCN 的两个关系变换矩阵 W_r
        self.w_rels = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])
        # 对齐公式 4: W_alpha 作用于拼接后的 2D 维度
        self.w_alpha = nn.Linear(hidden_dim * 2, 1)

    def forward(self, q, k, v, adj, s_mask):
        """
        q: H_i^{l-1} (当前节点上一层特征)
        k, v: H_j^l (历史节点当前层特征)
        adj: 窗口掩码
        s_mask: 说话人关系
        """
        batch_size, i_len, d = k.size()
        
        # 1. 关系感知特征变换 (公式 5 的核心)
        mask_expanded = s_mask.unsqueeze(-1).expand(-1, -1, d)
       # k_transformed = torch.where(mask_expanded == 1, self.w_rels[1](k), self.w_rels[0](k))
        v_transformed = torch.where(mask_expanded == 1, self.w_rels[1](v), self.w_rels[0](v))
        
        # 2. 计算注意力权重 (公式 4: 拼接 [||] 逻辑)
        q_exp = q.unsqueeze(1).expand(-1, i_len, -1) # (B, i, D)
        concat_qk = torch.cat([k, q_exp], dim=-1) # (B, i, 2D)
        
        # 注意：论文中通常直接线性变换，不加 tanh
        scores = self.w_alpha(concat_qk).squeeze(-1) # (B, i)
        
        # 3. Mask & Softmax
        scores = scores.masked_fill(adj == 0, -1e9)
        alpha = F.softmax(scores, dim=-1).unsqueeze(1) # (B, 1, i)
        
        # 4. 消息聚合
        M = torch.bmm(alpha, v_transformed).squeeze(1) # (B, D)
        return alpha, M


class DAGLayer(nn.Module):
    def __init__(self, config, hidden_dim, dropout, n_layers=2):
        super(DAGLayer, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.gather = nn.ModuleList([DAG_RGCN(hidden_dim) for _ in range(n_layers)])
        
        # 对齐公式 6 & 7: 定义双路 GRU
        self.grus_H = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.grus_M = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(n_layers)])
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, adj, s_mask):
        batch_size, num_utter, d = features.size()
        
        # 初始投影
        H0 = F.relu(self.fc1(features))
        H_layers = [H0]

        for l in range(self.n_layers):
            # --- 初始化第一个节点 (i=0) ---
            # 第一个节点没有历史 M，用全 0 向量代替
            M_zero = torch.zeros(batch_size, d).to(features.device)
            
            # 节点信息单元 (公式 6)
            H_tilde_0 = self.grus_H[l](H_layers[l][:, 0, :], M_zero).unsqueeze(1)
            # 上下文信息单元 (公式 7)
            C_0 = self.grus_M[l](M_zero, H_layers[l][:, 0, :]).unsqueeze(1)
            
            # 融合 (公式 8)
            H_current = H_tilde_0 + C_0 
            
            # --- 递归处理后续节点 (i > 0) ---
            for i in range(1, num_utter):
                # 1. 聚合历史 (公式 4 & 5)
                _, M = self.gather[l](
                    H_layers[l][:, i, :], 
                    H_current, 
                    H_current, 
                    adj[:, i, :i], 
                    s_mask[:, i, :i]
                )
                
                # 2. 双路门控更新 (公式 6, 7, 8)
                H_tilde_i = self.grus_H[l](H_layers[l][:, i, :], M).unsqueeze(1) # Eq. 6
                C_i = self.grus_M[l](M, H_layers[l][:, i, :]).unsqueeze(1)       # Eq. 7
                
                h_i = H_tilde_i + C_i # Eq. 8
                
                H_current = torch.cat((H_current, h_i), dim=1)
            
            H_layers.append(self.dropout(H_current))
            
        return H_layers[-1]

# GCN 模块
class GCN(nn.Module):
    def __init__(self, config, layer_num, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.config = config
        self.layer_list = nn.ModuleList()
        for i in range(layer_num):
            if i == layer_num - 1:
                self.layer_list.append(nn.Linear(hidden_dim, output_dim))
            elif i == 0:
                self.layer_list.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layer_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.gnn_dropout = nn.Dropout(dropout)
        self.gnn_activation = F.gelu

    def forward(self, x, mask, adj):
        D_hat = torch.diag_embed(torch.pow(torch.sum(adj, dim=-1), -1))
        if torch.isinf(D_hat).any():
            D_hat[torch.isinf(D_hat)] = 0.0
        adj = torch.matmul(D_hat, adj)

        x_mask = mask.unsqueeze(-1)
        for i, layer in enumerate(self.layer_list):
            if i != 0:
                x = self.gnn_dropout(x)
            x = torch.matmul(x, layer.weight.T) + layer.bias
            x = torch.matmul(adj, x)
            x = x * x_mask
            x = self.gnn_activation(x)

        return x



class BertWordPair(nn.Module):
    def __init__(self, config):
        super(BertWordPair, self).__init__()
        self.config = config 
        self.bert = AutoModel.from_pretrained(config.bert_path)
        
        bert_config = AutoConfig.from_pretrained(config.bert_path)
        bh = bert_config.hidden_size
        nhead = bert_config.num_attention_heads
        att_head_size = int(bh / nhead)

        self.config.loss_weight = {'ent': int(self.config.loss_w[0]), 'rel': int(self.config.loss_w[1]), 'pol': int(self.config.loss_w[2])}
        
        self.inner_dim = 256
        self.ent_dim = self.inner_dim * 4 * 4
        self.rel_dim = self.inner_dim * 4 * 3
        self.pol_dim = self.inner_dim * 4 * 4

        self.dense_all = nn.Linear(bert_config.hidden_size, self.ent_dim+self.rel_dim+self.pol_dim)
        
        self.dropout = nn.Dropout(config.dropout)

        self.interactLayer = InteractLayer(
            bert_config.hidden_size,
            bert_config.num_attention_heads,
            bert_config.hidden_dropout_prob,
            config
        )
                
        self.layernorm = nn.LayerNorm(bh, eps=1e-12)
        self.syngcn = GCN(config, config.gnn_layer_num, bh, bh, bh, config.gnn_dropout)

        self.semgcn = GCN(config, config.gnn_layer_num, bh, bh, bh, config.gnn_dropout)
        self.semantic_attention = MultiHeadAttention(bert_config.num_attention_heads, bh, att_head_size, att_head_size, bert_config.attention_probs_dropout_prob)

        # topk
        self.topK_select_layer = nn.Linear(bh, 1)
        self.utt_linear = nn.Linear(3*bh, bh)
        # --- 全局 GCN ---
        self.dsc_dag = DAGLayer(
            config=config, 
            hidden_dim=bh, 
            dropout=config.gnn_dropout, 
            n_layers=config.dscgnn_layer_num
        )
        self.global_layernorm = nn.LayerNorm(bh, eps=1e-12)
        # === HiRoPE 增强组件 (采纳建议1: 投影层) ===
        # 显式将 256 维压缩至 128 维，保留完整信息
        self.micro_proj = nn.Linear(self.inner_dim, self.inner_dim // 2)
        self.macro_proj = nn.Linear(self.inner_dim, self.inner_dim // 2)
        # === 调试标志位初始化 (Debug Flag) ===
        self.debug_has_printed = False 
        
        rope_status = getattr(self.config, 'use_rope', False)
        print("\n" + "#"*50)
        print(f"[DEBUG Init] DMIN Model Initialized (Version 3: Refined Expert).")
        print(f"[DEBUG Init] Config 'use_rope' status: {rope_status}")
        print("#"*50 + "\n")

    # --- HiRoPE Implementation Start (Expert Revised) ---

    def apply_rotary_pos_emb(self, x, cos, sin):
        return (x * cos) + (self.rotate_half(x) * sin)
    
    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # [FIX 1] 标准化频率计算，对齐 RoPE 论文定义
    def get_hirope_embedding(self, position_ids, dim, base_freq):
        # assert dim % 2 == 0, "RoPE dimension must be even" # 已经在外部保证
        half_dim = dim // 2
        
        # 使用标准的 RoPE 频率公式
        inv_freq = 1.0 / (base_freq ** (torch.arange(0, half_dim, dtype=torch.float, device=self.config.device) / half_dim))
        
        # position_ids: [len] -> [len, 1]
        # inv_freq: [half_dim] -> [1, half_dim]
        # angles: [len, half_dim]
        angles = torch.outer(position_ids.float(), inv_freq)
        
        # 拼接 cos/sin: [len, dim]
        cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
        sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
        
        return cos, sin
    
    # [FIX 2] 接收 q_token 和 q_utterance，实现真正的语义-层级对齐
    def get_instance_embedding_hirope(self, q_tok, q_utt, k_tok, k_utt, token_index, utterance_index, thread_length):
        seq_len, num_classes, hidden_size = q_tok.shape
        
        # [FIX 3] 安全检查
        assert hidden_size % 4 == 0, f"Hidden size ({hidden_size}) must be divisible by 4 for HiRoPE split (half must be even)."
        
        accu_index = [0] + list(accumulate(thread_length))
        logits = q_tok.new_zeros([seq_len, seq_len, num_classes])

        dim_split = hidden_size // 2
        
        for i in range(len(thread_length)):
            for j in range(len(thread_length)):
                rstart, rend = accu_index[i], accu_index[i+1]
                cstart, cend = accu_index[j], accu_index[j+1]

                # === 核心逻辑修改 ===
                # 我们从 q_tok 中切分出 micro (token) 语义
                # 我们从 q_utt 中切分出 macro (utterance) 语义
                # 这完美利用了 DMIN 的双重投影设计
                
                cur_q_tok = q_tok[rstart:rend] # [len_x, 6, 256]
                cur_q_utt = q_utt[rstart:rend] 
                cur_k_tok = k_tok[cstart:cend] # [len_y, 6, 256]
                cur_k_utt = k_utt[cstart:cend]

                t_idx_x, t_idx_y = token_index[rstart:rend], token_index[cstart:cend]
                u_idx_x, u_idx_y = utterance_index[rstart:rend], utterance_index[cstart:cend]

                # 相对距离翻转逻辑
                if i > 0 and i < j:
                    t_idx_x = -t_idx_x
                    u_idx_x = -u_idx_x
                if j > 0 and i > j:
                    t_idx_y = -t_idx_y
                    u_idx_y = -u_idx_y

                # === HiRoPE 维度组合 ===
                # [Refined FIX 1] 采纳建议1：使用 Linear 投影而不是切片
                qw_mic = self.micro_proj(cur_q_tok)
                kw_mic = self.micro_proj(cur_k_tok)

                qw_mac = self.macro_proj(cur_q_utt) # [..., 128]
                kw_mac = self.macro_proj(cur_k_utt)

                # 3. Micro Rotation (Base 10000)
                cos_mic_x, sin_mic_x = self.get_hirope_embedding(t_idx_x, dim_split, base_freq=10000)
                cos_mic_y, sin_mic_y = self.get_hirope_embedding(t_idx_y, dim_split, base_freq=10000)
                
                qw_mic = self.apply_rotary_pos_emb(qw_mic, cos_mic_x.unsqueeze(1), sin_mic_x.unsqueeze(1))
                kw_mic = self.apply_rotary_pos_emb(kw_mic, cos_mic_y.unsqueeze(1), sin_mic_y.unsqueeze(1))

                # 4. Macro Rotation (Base 100)
                cos_mac_x, sin_mac_x = self.get_hirope_embedding(u_idx_x, dim_split, base_freq=100) 
                cos_mac_y, sin_mac_y = self.get_hirope_embedding(u_idx_y, dim_split, base_freq=100)

                qw_mac = self.apply_rotary_pos_emb(qw_mac, cos_mac_x.unsqueeze(1), sin_mac_x.unsqueeze(1))
                kw_mac = self.apply_rotary_pos_emb(kw_mac, cos_mac_y.unsqueeze(1), sin_mac_y.unsqueeze(1))

                # 5. Fusion: 拼接 Micro 和 Macro
                cur_qw_rotated = torch.cat([qw_mic, qw_mac], dim=-1) # [len_x, 6, 256]
                cur_kw_rotated = torch.cat([kw_mic, kw_mac], dim=-1) # [len_y, 6, 256]

                pred_logits = torch.einsum('mhd,nhd->mnh', cur_qw_rotated, cur_kw_rotated).contiguous()
                logits[rstart:rend, cstart:cend] = pred_logits

        return logits

    # 更新接口，接收4个向量
    def get_ro_embedding(self, q_tok, q_utt, k_tok, k_utt, token_index, utterance_index, thread_lengths):
        logits = [] 
        batch_size = q_tok.shape[0]
        for i in range(batch_size):
            pred_logits = self.get_instance_embedding_hirope(
                q_tok[i], q_utt[i], k_tok[i], k_utt[i], # 传入全部4个语义向量
                token_index[i], utterance_index[i], 
                thread_lengths[i]
            )
            logits.append(pred_logits)
        logits = torch.stack(logits)
        return logits 

    # --- HiRoPE Implementation End ---
    
    def classify_matrix(self, kwargs, sequence_outputs, input_labels, masks, mat_name='ent'):
        utterance_index, token_index, thread_lengths = [kwargs[w] for w in ['utterance_index', 'token_index', 'thread_lengths']]

        outputs = torch.split(sequence_outputs, self.inner_dim * 4, dim=-1) 
        outputs = torch.stack(outputs, dim=-2) 
        
        all_splits = torch.split(outputs, self.inner_dim, dim=-1)
        
        # [FIX 2 - 提取] 获取 DMIN 原始计算的所有 4 个投影
        q_tok = all_splits[0]
        q_utt = all_splits[1] # 之前被忽略的 Utterance Query
        k_tok = all_splits[2]
        k_utt = all_splits[3] # 之前被忽略的 Utterance Key

        if getattr(self.config, 'use_rope', False) == True:
            # === DEBUG PRINT ===
            if not self.debug_has_printed:
                print("\n" + "="*60)
                print(f"🚀 [D-RoPE ACTIVATED - Expert Mode] Running classify_matrix for '{mat_name}'")
                print(f"   >>> Input Vector Strategy:")
                print(f"       Micro Semantic: Derived from q_token ({q_tok.shape})")
                print(f"       Macro Semantic: Derived from q_utterance ({q_utt.shape})")
                print(f"   >>> Splitting Strategy:")
                print(f"       First {self.inner_dim//2} dims of q_token -> Micro RoPE (Base 10000)")
                print(f"       First {self.inner_dim//2} dims of q_utterance -> Macro RoPE (Base 100)")
                print("="*60 + "\n")
                self.debug_has_printed = True
            # ===================
            
            # 传入 4 个向量，让 HiRoPE 内部去组合
            pred_logits = self.get_ro_embedding(q_tok, q_utt, k_tok, k_utt, token_index, utterance_index, thread_lengths)
        else:
            # 原版逻辑，只用 token 向量点积
            pred_logits = torch.einsum('bmhd,bnhd->bmnh', q_tok, k_tok).contiguous()

        nums = pred_logits.shape[-1]
        criterion = nn.CrossEntropyLoss(sequence_outputs.new_tensor([1.0] + [self.config.loss_weight[mat_name]] * (nums - 1)))
            
        active_loss = masks.view(-1) == 1
        active_logits = pred_logits.view(-1, pred_logits.shape[-1])[active_loss]
        active_labels = input_labels.view(-1)[active_loss]
        
        loss = criterion(active_logits, active_labels)
        
        return loss, pred_logits 

    # --- 以下为辅助函数保持不变 ---
    
    def merge_sentence(self, sequence_outputs, input_masks, dialogue_length):
        res = []
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            stack = []
            for j in range(s, e):
                lens = input_masks[j].sum()
                stack.append(sequence_outputs[j, :lens])
            res.append(torch.cat(stack))           
        new_res = sequence_outputs.new_zeros([len(res), max(map(len, res)), sequence_outputs.shape[-1]])
        for i, w in enumerate(res):
            new_res[i, :len(w)] = w
        return new_res 

    def root_merge_sentence(self, sequence_outputs, input_masks, dialogue_length, thread_lengths):
        if self.config.root_merge == 0:
            return self.merge_sentence(sequence_outputs, input_masks, dialogue_length)
        
        res = []
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            stack = []
            root_stack = []
            root_len = thread_lengths[i][0]
            for j in range(s, e):
                lens = input_masks[j].sum()
                root_stack.append(sequence_outputs[j, :root_len])
                stack.append(sequence_outputs[j, root_len:lens])

            root = torch.stack(root_stack).sum(0) / len(root_stack)

            stack = [root] + stack
            res.append(torch.cat(stack))  
        new_res = sequence_outputs.new_zeros([len(res), max(map(len, res)), sequence_outputs.shape[-1]])
        for i, w in enumerate(res):
            new_res[i, :len(w)] = w
        return new_res 
    
    def topk_aggregate(self, sentence_sequence_outputs, global_masks):
        batch_size, max_dialogue_length, hidden_size = sentence_sequence_outputs.shape
        batch_size, max_sentence_num, max_dialogue_length, _ = global_masks.shape

        sentence_lengths = global_masks.sum(dim=2).squeeze(-1) 

        split_sentences = []

        for i in range(batch_size):
            split_sentences.append([])
            for j in range(max_sentence_num):
                sentence_length = sentence_lengths[i, j]
                if sentence_length > 0:
                    start_index = (global_masks[i, j, :, :] == 1).nonzero()[0, 0].item()
                    end_index = int(start_index + sentence_length.item())
                    token_representation = sentence_sequence_outputs[i, start_index:end_index-1, :]
                    speaker_representation = sentence_sequence_outputs[i, end_index-1, :]
                    score = self.topK_select_layer(token_representation).squeeze(-1) / (sentence_length - 1)
                    k = int(self.config.topk*sentence_length)
                    k = k if k > 0 else 1

                    topk = torch.topk(score, k, dim=0, largest=True)[1]
                    score = torch.softmax(score[topk], dim=0)
                    token_representation = token_representation[topk]

                    token_representation = token_representation * score.unsqueeze(-1)

                    utt_representation = self.utt_linear(torch.cat((token_representation.mean(dim=0), token_representation.max(dim=0)[0], speaker_representation), dim=-1))
                     
                    split_sentences[i].append(utt_representation)
                else:
                    split_sentences[i].append(sentence_sequence_outputs.new_zeros([hidden_size]))
        split_sentences = torch.stack([torch.stack(bat) for bat in split_sentences], dim=0)

        return split_sentences

    def global_encoding(self, sentence_sequence_outputs, global_masks, dag_adj, dag_s_mask):
        # 1. 压缩词向量到句向量
        utterance_sequence = self.topk_aggregate(sentence_sequence_outputs, global_masks)      
        
        # 2. 调用正确的 tc-DAG 模块 (注意这里是 dsc_dag)
        # 传入 features, 窗口掩码 adj, 说话人掩码 s_mask
        global_outputs = self.dsc_dag(utterance_sequence, dag_adj, dag_s_mask)
        
        # 3. 残差连接与归一化
        global_outputs = self.global_layernorm(utterance_sequence + global_outputs)          

        return global_outputs
    
    def utterance2thread(self, sequence_outputs, thread_idxes, sentence_length, thread_lengths, merged_input_masks):
        thread_num, max_thread_len = merged_input_masks.shape
        
        thread_sequence_output = sequence_outputs.new_zeros([thread_num, max_thread_len, sequence_outputs.shape[-1]])
        thread_idx = 0
        for bat_idx, bat in enumerate(thread_idxes):
            for t_idx, thread in enumerate(bat):
                thread_list = []
                for s_idx, sent_idx in enumerate(thread):
                    thread_list.append(sequence_outputs[bat_idx, :sentence_length[bat_idx][sent_idx],:])
                thread_list = torch.cat(thread_list, dim=0)
                thread_sequence_output[thread_idx, :thread_list.shape[0], :] = thread_list
                thread_idx += 1
        
        return thread_sequence_output
    
    def forward(self, **kwargs):
        if self.config.merged_thread == 0:
            input_ids, input_masks, input_segments = [kwargs[w] for w in ['input_ids', 'input_masks', 'input_segments']]

        sentence_length, thread_idxes, merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length, merged_dialog_length, thread_lengths, adj_matrixes \
            = [kwargs[w] for w in ['sentence_length', 'thread_idxes', 'merged_input_ids', 'merged_input_masks', 'merged_input_segments', 'merged_sentence_length', 'merged_dialog_length', 'thread_lengths',  'adj_matrixes', ]]
        
        ent_matrix, rel_matrix, pol_matrix = [kwargs[w] for w in ['ent_matrix', 'rel_matrix', 'pol_matrix']]
        reply_masks, speaker_masks, thread_masks = [kwargs[w] for w in ['reply_masks', 'speaker_masks', 'thread_masks']]
        sentence_masks, full_masks, dialogue_length = [kwargs[w] for w in ['sentence_masks', 'full_masks', 'dialogue_length']]
        
        # 1. BERT Encoding
        if self.config.merged_thread == 1: 
            sequence_outputs = self.bert(merged_input_ids, token_type_ids=merged_input_segments, attention_mask=merged_input_masks)[0] 
            sentence_sequence_outputs = self.root_merge_sentence(sequence_outputs, merged_input_masks, merged_dialog_length, thread_lengths)
        else: 
            sequence_outputs = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[0] 
            sentence_sequence_outputs = self.merge_sentence(sequence_outputs, input_masks, dialogue_length)

        sentence_sequence_outputs = self.dropout(sentence_sequence_outputs)

        # 2. Local Encoding 
        if self.config.merged_thread == 1:
            syngcn_outputs = self.syngcn(sequence_outputs, merged_input_masks, adj_matrixes)
            syngcn_outputs = self.root_merge_sentence(syngcn_outputs, merged_input_masks, merged_dialog_length, thread_lengths)
        else:
            syngcn_outputs = self.syngcn(sequence_outputs, input_masks, adj_matrixes)
            syngcn_outputs = self.merge_sentence(syngcn_outputs, input_masks, dialogue_length)
        
        syngcn_outputs = self.dropout(syngcn_outputs)

        _, semantic_adj = self.semantic_attention(sequence_outputs, sequence_outputs, sequence_outputs)
        semantic_adj = semantic_adj.mean(dim=1)
        if self.config.merged_thread == 1:
            semgcn_output = self.semgcn(sequence_outputs, merged_input_masks, semantic_adj)
            semgcn_output = self.root_merge_sentence(semgcn_output, merged_input_masks, merged_dialog_length, thread_lengths)
        else:
            semgcn_output = self.semgcn(sequence_outputs, input_masks, semantic_adj)
            semgcn_output = self.merge_sentence(semgcn_output, input_masks, dialogue_length)
        semgcn_output = self.dropout(semgcn_output)
   
        sequence_outputs = self.layernorm(sentence_sequence_outputs+syngcn_outputs+semgcn_output)

        # 3. Global Encoding
        #if 'dag_adj' in kwargs:
        #    print(f"DEBUG: dag_adj shape: {kwargs['dag_adj'].shape}") 
            # 应该是 [Batch, Max_Sent, Max_Sent]
       #     print(f"DEBUG: dag_s_mask shape: {kwargs['dag_s_mask'].shape}")
        global_masks, dag_adj, dag_s_mask = [kwargs[w] for w in ['global_masks', 'dag_adj', 'dag_s_mask']]
        global_outputs = self.global_encoding(sentence_sequence_outputs, global_masks, dag_adj, dag_s_mask)
        
        # 4. Interaction
        thread_masks = thread_masks.bool().unsqueeze(1)
        sequence_outputs = self.interactLayer(sequence_outputs, global_outputs, thread_masks, sentence_length=sentence_length,)

        # 5. Decode
        sequence_outputs = self.dense_all(sequence_outputs)
        sequence_ent = sequence_outputs[:, :, :self.ent_dim]
        sequence_rel = sequence_outputs[:, :, self.ent_dim:self.ent_dim + self.rel_dim]
        sequence_pol = sequence_outputs[:, :, self.ent_dim + self.rel_dim:]
        
        ent_loss, ent_logit = self.classify_matrix(kwargs, sequence_ent, ent_matrix, sentence_masks, 'ent')
        rel_loss, rel_logit = self.classify_matrix(kwargs, sequence_rel, rel_matrix, full_masks, 'rel')
        pol_loss, pol_logit = self.classify_matrix(kwargs, sequence_pol, pol_matrix, full_masks, 'pol')

        total_loss = ent_loss + rel_loss + pol_loss

        return total_loss, [ent_loss, rel_loss, pol_loss], (ent_logit, rel_logit, pol_logit)