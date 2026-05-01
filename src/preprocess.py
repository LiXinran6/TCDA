
from src.utils import WordPair
import os
import re
import json

import numpy as np

from collections import defaultdict
from itertools import accumulate
from transformers import AutoTokenizer
from typing import List, Dict
from loguru import logger
from tqdm import tqdm

#核心任务是将原始的对话 JSON 数据转化为模型可直接计算的张量格式
class Preprocessor:
    def __init__(self, config):#接收配置对象 config。该对象包含了 BERT 路径、语言类型、隐藏层维度等关键超参数。
        self.config = config 
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)#加载预训练分词器（如 RoBERTa 或 BERT）。在 DMIN 中，分词器不仅用于切分单词，还决定了后续网格标注（Grid Tagging）的行数和列数。
        config.mask_id = self.tokenizer.mask_token_id#获取 [MASK] 标记的 ID。在 DMIN 的多粒度特征中，有时需要利用 Mask 机制来处理某些特征交互。
        config.cls_id = self.tokenizer.cls_token_id#获取 [CLS] 标记的 ID。在 DMIN 的词级增强编码器（CKEncoder）中，[CLS] 被视为句子的起始锚点，并在句法图中与其他词建立显式连接。
        config.pad_id = self.tokenizer.pad_token_id#获取 [PAD] 标记的 ID，用于将不同长度的对话填充对齐，支持批处理训练。
        self.wordpair = WordPair()#实例化 WordPair 类（来自 src/utils.py）。该类是处理网格标签（Grid Labels）的核心工具，负责四元组与网格坐标之间的相互转换。
        self.entity_dict = self.wordpair.entity_dic#获取预设的实体词典（包括 Target, Aspect, Opinion 等定义），用于后续在网格上生成标注真值。
    
    '''
    'OBIES' 逐字母拆解
    O (Outside)：表示该词不在任何实体中 。
    B (Begin)：表示该词是一个多词实体的起始部分 。
    I (Inside)：表示该词位于多词实体的中间部分 。
    E (End)：表示该词是一个多词实体的结束部分 。
    S (Single)：表示该实体仅由这一个词组成（单词实体） 。
    对于 Aspect（方面词）： 由于 asp_type: 'Aspect'，代码会生成：B-Aspect, I-Aspect, E-Aspect, S-Aspect, O
    对于 Opinion（意见词）： 它会结合情感极性（如 pos），生成：B-Opinion_pos, I-Opinion_pos, E-Opinion_pos, S-Opinion_pos
    '''

    def get_dict(self):#为 DMIN 的三个预测方格（Entity, Pair, Polarity）构建映射表（Dictionary），将抽象的标签名转换为模型可计算的整数索引。
        self.polarity_dict = self.config.polarity_dict#直接从配置文件中获取情感极性词典（例如 {'O': 0, 'pos': 1, 'neg': 2, 'neu': 3}）。

        # eg. BIESO {B-aspect:0, I-aspect:1 ...}构建 Aspect（方面词）边界标签词典
        self.aspect_dict = {}#初始化空字典
        for w in self.config.bio_mode:#遍历标注模式，通常是 B, I, E, S, O
            self.aspect_dict['{}{}'.format(w, '' if w == 'O' else '-' + self.config.asp_type)] = len(self.aspect_dict)#生成效果：{'O': 0, 'B-aspect': 1, 'I-aspect': 2, 'E-aspect': 3, 'S-aspect': 4}

        self.target_dict = {}#构建 Target（目标对象）边界标签词典，逻辑同上，只是将 asp_type 换成了 tgt_type（通常为 "target"）。
        for w in self.config.bio_mode:
            self.target_dict['{}{}'.format(w, '' if w == 'O' else '-' + self.config.tgt_type)] = len(self.target_dict)#生成效果：{'O': 0, 'B-target': 1, 'I-target': 2...}。

        self.opinion_dict = {'O': 0}#构建 Opinion（意见词）复合词典（含极性,实现了边界识别与情感分类的联合编码
        for p in self.polarity_dict:
            if p == 'O': continue
            for w in self.config.bio_mode[1:]:
                self.opinion_dict['{}-{}_{}'.format(w, self.config.opi_type, p)] = len(self.opinion_dict)#{'O': 0, 'B-opinion_pos': 1, ..., 'B-opinion_neg': 5, ...}
        
        self.relation_dict = {'O': 0, 'yes': 1}#构建 Relation（关系词典）与返回,这是 Pair Matrix 使用的词典，用于判定两个元素（如 Target 和 Aspect）是否有关联。
        return self.polarity_dict, self.target_dict, self.aspect_dict, self.opinion_dict, self.entity_dict, self.relation_dict#最后一次性返回六个词典供后续的索引转换（transform2indices）使用。
    
    #根据对话的结构（回复关系、发言人、线索）构建多视图掩码矩阵（Multi-view Masks）。这些矩阵决定了模型在处理词与词之间的关系时，哪些位置是“可见”的。
    def get_neighbor(self, utterance_spans, replies, max_length, speaker_ids, thread_nums):
        # utterance_mask = np.zeros([max_length, max_length], dtype=int)
        reply_mask = np.eye(max_length, dtype=int)#回复关系掩码，初始化为单位矩阵，即每个词默认与自己相连。
        for i, w in enumerate(replies):#遍历每句话及其对应的回复对象索引 w
            s1, e1 = utterance_spans[i]#获取当前句在全局序列中的起始和结束 Token 位置。
            s0, e0 = utterance_spans[w + (1 if w == -1 else 0)]#获取被回复句（父节点）的起始和结束位置。
            reply_mask[s0 : e0 + 1, s1 : e1 + 1] = 1#将当前句的所有 Token 与被回复句的所有 Token 进行双向打通（即矩阵中这两个区域全设为 1）。同时确保句内 Token 也是互联的
            reply_mask[s1 : e1 + 1, s0 : e0 + 1] = 1
            reply_mask[s0 : e0 + 1, s0 : e0 + 1] = 1
            reply_mask[s1 : e1 + 1, s1 : e1 + 1] = 1
        #这部分代码建立了基于“说话人身份”的词级关联 。
        speaker_mask = np.zeros([max_length, max_length], dtype=int)#初始化为全零矩阵
        for i, idx in enumerate(speaker_ids):#双重循环遍历 speaker_ids: 寻找对话中所有由同一个人说出的句子。
            # utterance_ids = [j for j, w in enumerate(speaker_ids) if w == idx]
            s0, e0 = utterance_spans[i]
            for j, idx1 in enumerate(speaker_ids):
                if idx != idx1: continue#如果不是同一个发言人，则不建立联系
                s1, e1 = utterance_spans[j] #将同一发言人在不同轮次说出的所有 Token 进行两两互联。
                speaker_mask[s0 : e0 + 1, s1 : e1 + 1] = 1
                speaker_mask[s1 : e1 + 1, s0 : e0 + 1] = 1
                speaker_mask[s0 : e0 + 1, s0 : e0 + 1] = 1
                speaker_mask[s1 : e1 + 1, s1 : e1 + 1] = 1
        
        thread_mask = np.eye(max_length, dtype=int)#线索掩码,实现了 DMIN 论文中提到的线索内（In-thread）信息限制
        thread_ends = accumulate(thread_nums)#accumulate: 计算累加和。thread_ends 变成 [1, 4, 6]。注意，这里的thread_nums已经是连续的了，经过处理后的
        thread_spans = [(w - z, w) for w, z in zip(thread_ends, thread_nums)]#根据 thread_nums（每个线索包含的句子数）计算出每个线索在句子序列中的起止范围，计算每个线索在句子列表中的起止索引。结果为 [(0, 1), (1, 4), (4, 6)]。
        for i, (s, e) in enumerate(thread_spans):#线索内“根节点注入” (Root Injection)
            if i == 0: continue
            head_start, head_end = utterance_spans[0]#utterance_spans[0] 指的是对话中第一句话（Root Utterance）在整个长序列中的起止坐标。
            thread_mask[head_start : head_end + 1, head_start : head_end + 1] = 1#为对话的第一句话（根话语，Root Utterance）构建“全自环”连接
            for j in range(s, e):#外层循环遍历线索 i 中的每一句话 j。它强行让线索中的每一句话与对话的第一句话（head）互联。
                s0, e0 = utterance_spans[j]#无论你是第几个 Thread，无论你在讨论什么支线，代码都会在 thread_mask 里划出一个长方形，把你当前句（j）和第一句话（head）双向打通
                thread_mask[s0:e0 + 1, head_start:head_end+1] = 1
                thread_mask[head_start:head_end+1, s0:e0 + 1] = 1
                for k in range(s, e):#在当前的 Thread 范围 (s, e) 内，所有的句子两两之间全部连通，因此，第一句话和某个线索的所有句子全部都是两两打通的
                    s1, e1 = utterance_spans[k]
                    thread_mask[s0 + 1 : e0, s1 + 1: e1] = 1
                    thread_mask[s1 + 1 : e1, s1 + 1: e1] = 1
                    thread_mask[s0 + 1 : e0, s0 + 1: e0] = 1
                    thread_mask[s1 + 1 : e1, s0 + 1: e0] = 1

        return reply_mask.tolist(), speaker_mask.tolist(), thread_mask.tolist()
    
    #将混乱的、交织的对话历史，重组成多条清晰的、以根节点（Root）为起点的“逻辑线索（Thread）”路径

    def find_utterance_index(self, replies, sentence_lengths):
        utterance_collections = [i for i, w in enumerate(replies) if w == 0]#replies[i] == 0 通常代表这句话回复的是“根节点”（Root）。这意味着这句话是一个新线索的开始。找出所有新线索开头的句子索引。比如 [0, 5, 10] 表示第 0、5、10 句分别开启了三个讨论分支。
        zero_index = utterance_collections[1]#重置相对索引 (Relative Indexing)将原始的绝对句子索引转换为线索（Thread）内的相对偏移索引
        for i in range(len(replies)):
            if i < zero_index: continue
            if replies[i] == 0:#如果当前句子标志着一个新线索的开始（在 DMIN 数据格式中，replies 为 0 代表它回复的是根节点，即开启了新支线）。
                zero_index = i#核心动作：重置基准点。一旦发现新线索，就把当前的物理位置 i 记作新的“零点”
            replies[i] = (i - zero_index)#用当前的物理索引 i 减去当前线索的起点 zero_index。

        sentence_index = [w + 1 for w in replies]#将上一步计算出的相对偏移 replies（如 0, 1, 2...）统一加 1。

        utterance_index = [[w] * z for w, z in zip(sentence_index, sentence_lengths)]#这一步是在做**“属性广播”**。它把属于句子的“序号属性”分配给了该句名下的每一个词。如果第 1 句话有 3 个词，第 2 句话有 2 个词，结果就是 [[1, 1, 1], [2, 2]]
        utterance_index = [w for line in utterance_index for w in line]#将上面的嵌套列表“压平”（Flatten）成一个一维列表。[[1, 1, 1], [2, 2]] $\rightarrow$ [1, 1, 1, 2, 2]

        #**“从原始时间流到树状逻辑流”**的结构转换
        token_index = [list(range(sentence_lengths[0]))]#为第一句话（Root）生成位置索引。如果第一句长 5，则生成 [[0, 1, 2, 3, 4]]
        lens = len(token_index[0])#记录第一句话的长度，作为后续线索累加的基础
        for i, w in enumerate(sentence_lengths):#遍历所有句子的长度。
            if i == 0: continue#跳过已经处理过的第一句
            if sentence_index[i] == 1:#核心逻辑：检查当前句子是否是一个新线索的开始。
                distance = lens#如果是新线索的开始，它的“位置起点”要重置为第一句话（Root）的结束位置。这保证了所有线索都紧跟在 Root 后面计数。
            token_index += [list(range(distance, distance + w))]#为当前句子生成一段连续的索引。例如 distance 是 5，句子长 3，则生成 [5, 6, 7]。
            distance += w#更新当前线索的累计长度，供下一句使用。
        token_index = [w for line in token_index for w in line] #将嵌套列表展平。

        utterance_collections = np.split(sentence_index, utterance_collections)#利用之前找出的“新线索起点” utterance_collections（如 [0, 5, 10]），将长长的句子索引数组切成一个个小数组。

        thread_nums = list(map(len, utterance_collections))#计算每个切分出来的数组长度它代表了每个线索包含多少个句子。比如结果是 [5, 5]，说明对话被切成了两个线索，每个线索 5 句话。
        thread_ranges = [0] + list(accumulate(thread_nums))#计算句子的累加边界。如 [0, 5, 10]
        thread_lengths = [sum(sentence_lengths[thread_ranges[i]:thread_ranges[i+1]]) for i in range(len(thread_ranges)-1)]#根据句子边界，去 sentence_lengths（每个句子的词数）里把属于同一个线索的词数加起来。代表了每个线索包含多少个 Token
        #这一段是在为“谁回复了谁”建立一个简洁的查询表。
        sent_idx2reply_idx = defaultdict(int)
        for sent_idx, reply in enumerate(replies):
            if reply == -1:
                sent_idx2reply_idx[sent_idx] = 0
            elif reply == 0:
                sent_idx2reply_idx[sent_idx] = 0
            else:
                sent_idx2reply_idx[sent_idx] = last_reply_idx
            last_reply_idx = sent_idx

        return utterance_index, token_index, thread_lengths, thread_nums, sent_idx2reply_idx
    #utterance_index词所属的句号，token_index线索内绝对位置，thread_lengths线索总词数，thread_nums线索句子数，sent_idx2reply_idx因果映射表
    #从已经抽取的完整四元组（Full Triplets）中，提取出两两之间的二元关联对（Pairs）。
    def get_pair(self, full_triplets):
        pairs = {'ta': set(), 'ao': set(), 'to': set()}
        for i in range(len(full_triplets)):
            st, et, sa, ea, so, eo, p = full_triplets[i][:7]
            if st != -1 and sa != -1:
                pairs['ta'].add((st, et, sa, ea))

            if st != -1 and so != -1:
                pairs['to'].add((st, et, so, eo))

            if sa != -1 and eo != -1:
                pairs['ao'].add((sa, ea, so, eo))

        return pairs
    ##将输入的情感极性（Polarity）字符串进行归一化处理。
    def transfer_polarity(self, pol):
        res = {'pos': 'pos', 'neg': 'neg'}
        return res.get(pol, 'other')
    
    #读取原始 JSON 文件，并启动“分词-对齐-解析”这一系列复杂的流水线。
    def read_data(self, mode):
        """
        Read a JSON file, tokenize using BERT, and realign the indices of the original elements according to the tokenization results.
        """

        path = os.path.join(self.config.json_path, '{}.json'.format(mode))
        if self.config.testset_name is not None and 'test' in mode:
            path = os.path.join(self.config.json_path, '{}'.format(self.config.testset_name))
        print("dataset path: ", path)

        if not os.path.exists(path):
            raise FileNotFoundError('File {} not found! Please check your input and data path.'.format(path))

        content = json.load(open(path, 'r', encoding='utf-8'))
        res = []
        for line in tqdm(content, desc='Processing dialogues for {}'.format(mode)):
            new_dialog = self.parse_dialogue(line, mode)#它会在这里调用你之前看过的 find_utterance_index、get_neighbor 等所有逻辑，把一个原始的 JSON 字典变成一个带有各种 DAG 索引、Mask 矩阵和 BERT Token 的“超级字典”
            res.append(new_dialog)
        return res#列表中的每一个元素都是一个经过深度解析的 字典（Dictionary）
    
    def check_text(self, tokenized_text, source_text):#确保 BERT/RoBERTa 分词（Tokenization）后的内容与原始文本（Source Text）在逻辑上是一致的
        if self.config.bert_path in ['roberta-large', 'roberta-base']:
            t0 = tokenized_text.lower()
            roberta_chars = 'â ī ¥ Ġ ð ł ĺ ħ Ł ŀ į Ŀ Į ĵ © ĵ ĳ ¶ ã'.split()
            unused = [self.config.unk, '##']
            if self.config.bert_path in ['roberta-large', 'roberta-base']:
                unused += roberta_chars
            for u in unused:
                t0 = t0.replace(u.lower(), '')
            t1 = source_text.replace(' ', '').lower()
            for k in self.config.unkown_tokens:
                t1 = t1.replace(k, '')
            if self.config.bert_path in ['roberta-large', 'roberta-base']:
                t1 = t1.replace('×', '').replace('≥', '')
            if t0 != t1:
                logger.info(t1 + '||' + t1)
                logger.info(tokenized_text + '||' + source_text)
                t2 = t0
                for u in unused:
                    t2 = t2.replace(u, '')
                raise AssertionError("--{}-- != --{}--".format(t0, t1))
            return t0 == t1

        t0 = tokenized_text.replace('##', '').replace(self.config.unk, '').lower()
        t1 = source_text.replace(' ', '').lower()
        for k in self.config.unkown_tokens:
            t1 = t1.replace(k, '')
        if t0 != t1:
            logger.info(t1 + '||' + t1)
            logger.info(tokenized_text + '||' + source_text)
            raise AssertionError("{} != {}".format(t0, t1))
        return t0 == t1
    
    #在对话 ABSA 中，标注好的 Target、Aspect（比如“耳机”是第 5-6 个词）是基于原始文本的。但模型输入的是 BERT 切分后的 Sub-tokens (Pieces)。这个函数的作用就是把所有标注的索引，从“原始单词坐标”精准地平移到“BERT 碎片坐标”上。
    #它的输入是原始的、基于单词（Word）标注的对话数据，输出是模型可直接读取的、基于分词碎片（Piece/Sub-token）对齐后的结构化数据。
    def parse_dialogue(self, dialogue, mode):
        # get the list of sentences in the dialogue
        sentences= dialogue['sentences']

        # align_index_with_list: align the index of the original elements according to the tokenization results
        # eg. pieces2words = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4]
        if 'dep' in mode:
            piece_dep = dialogue['piece_dep']
            new_sentences = piece_dep['pieces'] 
            targets, aspects, opinions, triplets, pieces2words = [piece_dep[w] for w in ['targets', 'aspects', 'opinions', 'triplets', 'dep_piece2ori_token']]
            # thread_piece = piece_dep['thread_pieces'] 
            # dialogue['thread_piece'] = thread_piece
        else:
            new_sentences, pieces2words = self.align_index_with_list(sentences)

            word2pieces = defaultdict(list) 
            for p, w in enumerate(pieces2words):
                word2pieces[w].append(p)


            # get target, aspect and opinion respectively, and align to the new index

            # if 'train' not in mode and 'valid' not in mode:
            #     return dialogue
            targets, aspects, opinions = [dialogue[w] for w in ['targets', 'aspects', 'opinions']]
            targets = [(word2pieces[x][0], word2pieces[y-1][-1] + 1, z) for x, y, z in targets]
            aspects = [(word2pieces[x][0], word2pieces[y-1][-1] + 1, z) for x, y, z in aspects]
            opinions = [(word2pieces[x][0], word2pieces[y-1][-1] + 1, z, self.transfer_polarity(w)) for x, y, z, w in opinions]
            
            # polarity transfer and index transfer
            triplets = []
            for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in dialogue['triplets']:
                polarity = self.transfer_polarity(polarity)
                nts, nas, nos = [word2pieces[w][0] if w != -1 else -1 for w in [t_s, a_s, o_s]]
                nte, nae, noe = [word2pieces[w - 1][-1] + 1 if w != -1 else -1 for w in [t_e, a_e, o_e]]
                triplets.append((nts, nte, nas, nae, nos, noe, polarity, t_t, a_t, o_t))



        # Confirm the index again
        # Flatten the two-dimensional list and put the entire dialogue in a list
        news = [w for line in new_sentences for w in line]
        for ts, te, t_t in targets:
            assert self.check_text(''.join(news[ts:te]), t_t)
        for ts, te, t_t in aspects:
            assert self.check_text(''.join(news[ts:te]), t_t)
        for ts, te, t_t,_ in opinions:
            assert self.check_text(''.join(news[ts:te]), t_t)
        for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in triplets:
            self.check_text(''.join(news[t_s:t_e]), t_t)
            self.check_text(''.join(news[a_s:a_e]), a_t) or a_s == -1
            if not self.check_text(''.join(news[o_s:o_e]), o_t) and o_s != -1:
                logger.info(''.join(news[o_s:o_e]) + '||' + o_t)
            self.check_text(''.join(news[o_s:o_e]), o_t) or o_s == -1

        # Put the elements into the dialogue object after converting the elements to the new index
        dialogue['sentences'] = new_sentences
        dialogue['targets'], dialogue['aspects'], dialogue['opinions'] = targets, aspects, opinions
        dialogue['triplets'] = triplets
        dialogue['pieces2words'] = pieces2words
        # DO tokenized
        return dialogue 

    #这段代码是 parse_dialogue 中最核心的“翻译工具”，它的作用是执行 BERT 分词 并建立 子词（Piece）与原始单词（Word）之间的映射关系。
    #[0, 0, 1, 2, 2]  含义：索引 0 和 1 的 Piece 属于原词 0；索引 2 属于原词 1；索引 3 和 4 属于原词 2。
    def align_index_with_list(self, sentences):
        """_summary_
        align the index of the original elements according to the tokenization results
        Args:
            sentences (_type_): List<str>
            e.g., xiao mi 12x is my favorite
        """
        pieces2word = []
        word_num = 0
        all_pieces = []
        for sentence in sentences:
            sentence = sentence.split()
            tokens = [self.tokenizer.tokenize(w) for w in sentence]
            cur_line = []
            for token in tokens:
                for piece in token:
                    pieces2word.append(word_num)
                word_num += 1
                cur_line += token
            all_pieces.append(cur_line)
        
        return all_pieces, pieces2word
    
    #align_index_with_list 的精简版本，专门用于处理**语法依存（Dependency Parsing）**相关的索引对齐。它不负责切分句子，而是直接接收已经分好词的原始 Token 列表（ori_tokens），并返回从 Sub-tokens (Pieces) 到 Original Tokens (Words) 的映射表。
    def align_index_with_list_dep(self, ori_tokens):
        pieces2word = []
        word_num = 0
        tokens = [self.tokenizer.tokenize(w) for w in ori_tokens]
        for token in tokens:
            for piece in token:
                pieces2word.append(word_num)
            word_num += 1
        
        return  pieces2word
    #在“原始字符（Character）与 Sub-token”**之间做对齐，生成一个巨大的字典 char2token，它能告诉你：原始文本中第 $N$ 个字符，对应分词后序列中的第 $M$ 个 Token。
    def align_index(self, sentences):
        res, char2token = [], {}
        source_lens, token_lens = 0, 0
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if self.config.bert_path in ['roberta-large', 'bert-base-uncased']:
                c2t, tokens = self.alignment_roberta(sentence, tokens)
            else:
                c2t, tokens = self.alignment(sentence, tokens)
            res.append(tokens)
            for k, v in c2t.items():
                char2token[k + source_lens] = v + token_lens
            source_lens, token_lens = source_lens + len(sentence) + 1, token_lens + len(tokens)

        return res, char2token
    #基于双指针算法（Two-Pointer）实现原始字符（Character）到模型碎片（Token/Piece）的硬对齐
    def alignment(self, source_sequence, tokenized_sequence: List[str], align_type: str = 'one2many') -> Dict:
        """[summary]
        # this is a function that to align sequcences  that before tokenized and after.
        Parameters
        ----------
        source_sequence : [type]
            this is the original sequence, whose type either can be str or list
        tokenized_sequence : List[str]
            this is the tokenized sequcen, which is a list of tokens.
        index_type : str, optional, default: str
            this indicate whether source_sequence is str or list, by default 'str'
        align_type : str, optional, default: one2many
            there may be several kinds of tokenizer style, 
            one2many: one word in source sequence can be split into multiple tokens 
            many2one: many word in source sequence will be merged into one token
            many2many: both contains one2many and many2one in a sequence, this is the most complicated situation.
        
        useage:
        source_sequence = "Here, we investigate the structure and dissociation process of interfacial water"
        tokenized_sequence = ['here', ',', 'we', 'investigate', 'the', 'structure', 'and', 'di', '##sso', '##ciation', 'process', 'of', 'inter', '##fa', '##cial', 'water']
        char2token = alignment(source_sequence, tokenized_sequence)
        print(char2token)
        for c, t in char2token.items():
            print(source_sequence[c], tokenized_sequence[t])
        """
        char2token = {}
        if isinstance(source_sequence, str) and align_type == 'one2many':
            source_sequence = source_sequence.lower()
            i, j = 0, 0
            while i < len(source_sequence) and j < len(tokenized_sequence):
                cur_token, length = tokenized_sequence[j], len(tokenized_sequence[j])
                if source_sequence[i] == ' ':
                    i += 1
                elif source_sequence[i: i + length] == cur_token:
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
                elif tokenized_sequence[j] == self.config.unk:
                    lens = 1
                    if j + 1 == len(tokenized_sequence):
                        lens = len(source_sequence) - i
                    else:
                        while i + lens < len(source_sequence):
                            if source_sequence[i + lens] == tokenized_sequence[j + 1].strip('#')[0] or tokenized_sequence[j+1] == self.config.unk:
                                break
                            lens += 1
                    new_token = self.repack_unknow(source_sequence[i:i+lens])
                    tokenized_sequence = tokenized_sequence[:j] + new_token + tokenized_sequence[j+1:]
                    if tokenized_sequence[j] == self.config.unk:
                        char2token[i] = j
                        i += 1
                        j += 1
                else:
                    assert tokenized_sequence[j].startswith('#')
                    length = len(tokenized_sequence[j].lstrip('#'))
                    assert source_sequence[i: i + length] == tokenized_sequence[j].lstrip('#')
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
        return char2token, tokenized_sequence
    
    #这段代码是专门为 RoBERTa 分词器定制的字符级对齐函数。它在逻辑上与你之前看到的 alignment 非常相似，但针对 RoBERTa 的 BPE（Byte-Pair Encoding） 特性做了关键的适配。
    def alignment_roberta(self, source_sequence, tokenized_sequence: List[str]) -> Dict:
        # For English dataset
        char2token = {}
        if isinstance(source_sequence, str):
            source_sequence = source_sequence.lower()
            i, j = 0, 0
            while i < len(source_sequence) and j < len(tokenized_sequence):
                cur_token, length = tokenized_sequence[j], len(tokenized_sequence[j].strip('Ġ'))
                if source_sequence[i] == ' ':
                    i += 1
                elif source_sequence[i: i + length].lower() == cur_token.strip('Ġ').lower():
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
                elif tokenized_sequence[j] == self.config.unk:
                    lens = 1
                    if j + 1 == len(tokenized_sequence):
                        lens = len(source_sequence) - i
                    else:
                        while i + lens < len(source_sequence):
                            if source_sequence[i + lens] == tokenized_sequence[j + 1].strip('#')[0] or tokenized_sequence[j+1] == self.config.unk:
                                if tokenized_sequence[j+1].strip('#')[0] == 'i' and j + 1 < len(tokenized_sequence) and len(tokenized_sequence[j+1].strip()) > 1:
                                    if i + lens + 1 < len(source_sequence) and source_sequence[i+lens+1] == tokenized_sequence[j+1].strip('#')[1]: 
                                        break
                                else:
                                    break
                            lens += 1
                    new_token = self.repack_unknow(source_sequence[i:i+lens])
                    tokenized_sequence = tokenized_sequence[:j] + new_token + tokenized_sequence[j+1:]
                    if tokenized_sequence[j] == self.config.unk:
                        char2token[i] = j
                        i += 1
                        j += 1
                else:
                    assert tokenized_sequence[j].startswith('#')
                    length = len(tokenized_sequence[j].lstrip('#'))
                    assert source_sequence[i: i + length] == tokenized_sequence[j].lstrip('#')
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
        return char2token, tokenized_sequence
    #解决 BERT 原生分词器在遇到特殊字符（如 Emoji、特殊符号）与正常字符混合**时的“吞词”问题。
    def repack_unknow(self, source_sequence):
        '''
        # sentence='🍎12💩', Bert can't recognize two contiguous emojis, so it recognizes the whole as '[UNK]'
        # We need to manually split it, recognize the words that are not in the bert vocabulary as UNK, 
        and let BERT re-segment the parts that can be recognized, such as numbers
        # The above example processing result is: ['[UNK]', '12', '[UNK]']
        '''
        lst = list(re.finditer('|'.join(self.config.unkown_tokens), source_sequence))
        start, i = 0, 0
        new_tokens = []
        while i < len(lst):
            s, e = lst[i].span()
            if start < s:
                token = self.tokenizer.tokenize(source_sequence[start:s]) 
                new_tokens += token
                start = s
            else:
                new_tokens.append(self.config.unk)
                start = e
            i += 1
        if start < len(source_sequence):
            token = self.tokenizer.tokenize(source_sequence[start:]) 
            new_tokens += token
        return new_tokens
    #将散落在列表里的独立句子，按照之前计算好的线索（Thread）逻辑，物理性地缝合在一起
    #接收的是已经分好词、但还处于“散装”状态的句子
    #返回的是一个元组，最核心的是拼接后的 ID 列表
    #merged_input_ids 就是一个列表（List），列表里的每一个元素（Item）都是一条完整线索内部句子的物理拼接。
    def merge_same_thread(self, input_ids, input_masks, input_segments,sentence_length, utterance_index, thread_length):
         
        merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length = [], [], [], []
        
        start_idx = 0
        idx_pairs = []
        j = 0
        sentence_len = sentence_length
        for tl in thread_length:#找到哪些句子属于同一个 Thread，利用 thread_length（每个线索的总词数）作为刻度尺，在句子列表里“切块”，存储了每个线索包含的句子区间（如 (0, 3) 表示前三句是一个线索）
            all_len = 0
            while (j < len(sentence_len)):
                all_len += sentence_len[j]
                j += 1 
                if all_len == tl:
                    idx_pairs.append((start_idx, j))
                    start_idx = j
                    break
        merged_input_ids = [ [ a  for i in range(start, end) for a in input_ids[i]] for (start, end) in idx_pairs]
        # non speaker position in thread
        nonspeaker_token_positions = []
        for (start, end) in idx_pairs:
            cur_thread_len = 0
            ns_p = []
            for i in range(start, end):
                ns_p.append([cur_thread_len+3, cur_thread_len+len(input_ids[i])])
                cur_thread_len+=len(input_ids[i])
            nonspeaker_token_positions.append(ns_p)
        root_n_sp = nonspeaker_token_positions[0][0]
        for i in range(1,len(nonspeaker_token_positions)):
            for j in range(len(nonspeaker_token_positions[i])):
                nonspeaker_token_positions[i][j] = [nonspeaker_token_positions[i][j][0]+root_n_sp[1], nonspeaker_token_positions[i][j][1]+root_n_sp[1]]
            nonspeaker_token_positions[i].insert(0, root_n_sp)
        nonspeaker_token_positions.pop(0)
        
        if 'roberta' in  self.config.bert_path:
            merged_input_segments = [ [ 0 for i in range(start, end) for a in input_segments[i]] for (start, end) in idx_pairs]
        else:
            merged_input_segments = [ [ 0 if i==start else 1 for i in range(start, end) for a in input_segments[i]] for (start, end) in idx_pairs]
        merged_input_masks = [ [ a for i in range(start, end) for a in input_masks[i]] for (start, end) in idx_pairs]
        merged_sentence_length = [ sum([ sentence_length[i] for i in range(start, end) ]) for (start, end) in idx_pairs]

        if self.config.root_merge==1:
            root_merged_input_ids = [merged_input_ids[0]+merged_input_ids[i] for i in range(1, len(merged_input_ids))]
            root_merged_input_masks = [merged_input_masks[0]+merged_input_masks[i] for i in range(1, len(merged_input_masks))]
            root_merged_input_segments = [merged_input_segments[0]+merged_input_segments[i] for i in range(1, len(merged_input_segments))]
            root_merged_sentence_length = [merged_sentence_length[0]+merged_sentence_length[i] for i in range(1, len(merged_sentence_length))]
            return root_merged_input_ids, root_merged_input_masks, root_merged_input_segments, root_merged_sentence_length, nonspeaker_token_positions
    
        return merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length, nonspeaker_token_positions
    
#Graph（图）构建阶段的核心逻辑。它的任务是手动定义 Adjacency Matrix（邻接矩阵） 里的边，从而决定信息在不同 Token 之间如何流动
    def link_adj(self, adj_matrix, cls_list, sep_list, root_list, piece_list, head_list):
        # piece 让属于同一个词的碎片两两之间（顺序）互相打通
        for pie in piece_list:
                for i in range(len(pie)-1):
                    adj_matrix[pie[i], pie[i+1]] = 1.0
                    adj_matrix[pie[i+1], pie[i]] = 1.0

        # root连接对象：每个句子的 核心词（Root Words）在句子内部，把重要的关键词串起来。跨句子连接：让前一个句子的结尾核心词指向后一个句子的开头核心词。
        for sent_idx in range(len(root_list)):
            for r_idx in range(len(root_list[sent_idx])):
                if r_idx+1 < len(root_list[sent_idx]):
                    adj_matrix[root_list[sent_idx][r_idx][0], root_list[sent_idx][r_idx+1][0]] = 1.0
                    adj_matrix[root_list[sent_idx][r_idx+1][0], root_list[sent_idx][r_idx][0]] = 1.0

            if sent_idx+1 < len(root_list): # cross sentence
                adj_matrix[root_list[sent_idx][-1][0], root_list[sent_idx+1][0][0]] = 1.0


        for i in range(len(sep_list)):#连接对象：每句话开头的 [CLS] 和结尾的 [SEP]。：让一句话的“开头”和“结尾”形成闭环。这有助于模型捕捉句子的全局表示，让 [CLS] 能够快速收集到整句话的汇总信息。
                adj_matrix[sep_list[i], cls_list[i]] = 1.0
                adj_matrix[cls_list[i], sep_list[i]] = 1.0
            
        return adj_matrix
#根据语法依存关系和句子的逻辑结构，在邻接矩阵中“勾画”出所有可能的语义流动路径。
    def get_adj_matrix(self, deprel, head_list):
        n = len(head_list)
        adj_matrix = np.array([[0.0] * n for i in range(n)])

        # 1. self-link edge自环边 (Self-link Edge) —— 保证自我信息
        for i in range(n):
            adj_matrix[i][i] = 1.0
        # 2. dependent edge语法依存边 (Dependent Edge) —— 建立句法关联  利用外部解析器得到的 head_list（谁是谁的爸爸），在对应的两个 Token 之间连线。
        for i in range(n):
            j = head_list[i]
            if j >= 0 and deprel[i] not in ['piece', 'ROOT', 'SENT_BEGIN', 'SENT_END']:
                adj_matrix[i][j] = 1.0
                adj_matrix[j][i] = 1.0  # make symmetric
        # 3. sentence edge结构化边 (Sentence Edge) —— 梳理对话脉络，识别 Token 的身份
        cls_list, sep_list, root_list, piece_list = [], [], defaultdict(list), []
        utterance_index = 0
        i = 0
        while i < len(deprel):
            dep = deprel[i]
            if dep == 'SENT_BEGIN':
                cls_list.append(i)
            elif dep == 'SENT_END':
                sep_list.append(i)
                utterance_index += 1
            elif dep == 'ROOT':
                roots = []
                while i < len(deprel) and (deprel[i] == 'ROOT' or deprel[i] == 'piece'):
                    roots.append(i)
                    i += 1
                root_list[utterance_index].append(roots)
                continue
            elif dep == 'piece':
                pieces = [i-1]
                while i < len(deprel) and deprel[i] == 'piece':
                    pieces.append(i)
                    i += 1
                piece_list.append(pieces)
                continue
            i += 1

        adj_matrix = self.link_adj(adj_matrix, cls_list, sep_list, root_list, piece_list, head_list)
        

        return adj_matrix
    
    def get_tc_dag_metadata2(self, speakers, thread_nums, w_size):
        """
        实现 TC-DAG 拓扑约束 (修改版):
        1. 线索内序列连接: 严格限制在线索内部回溯
        2. 全局根节点可达: 若线索内回溯未填满窗口，允许连接到根节点 u0
        """
        n = len(speakers)
        adj = np.zeros((n, n), dtype=np.float32)
        s_mask = np.zeros((n, n), dtype=np.int64)

        # --- 预处理：构建线索映射 ---
        thread_ends = list(accumulate(thread_nums))
        thread_starts = [w - z for w, z in zip(thread_ends, thread_nums)]
        
        sent2thread = {}
        for t_idx, (start, end) in enumerate(zip(thread_starts, thread_ends)):
            for s_idx in range(start, end):
                sent2thread[s_idx] = t_idx

        for i in range(n):
            t_i = sent2thread[i]
            start_of_thread = thread_starts[t_i]
            
            # 计数器：当前节点看到的同说话人数量
            cnt = 0
            
            # --- 阶段 1: 线索内回溯 (Intra-Thread Search) ---
            # 范围：从 i-1 回退到当前线索的起始位置 start_of_thread
            for j in range(i - 1, start_of_thread - 1, -1):
                adj[i, j] = 1.0
                s_mask[i, j] = 1 if speakers[j] == speakers[i] else 0
                
                if speakers[j] == speakers[i]:
                    cnt += 1
                    # 如果窗口已满，立即停止所有搜索（包括对根节点的搜索）
                    if cnt == w_size:
                        break

            # --- 阶段 2: 全局根节点连接 (Global Root Connection) ---
            # 逻辑：
            # 1. 必须是其他线索 (t_i > 0)。因为如果 t_i == 0，上面的循环已经自然包含了 u0 (索引0)。
            # 2. 必须还有窗口余额 (cnt < w_size)。
            if t_i > 0 and cnt < w_size:
                adj[i, 0] = 1.0
                s_mask[i, 0] = 1 if speakers[i] == speakers[0] else 0
                
                # (可选) 如果需要严格统计 u0 的说话人计数，可以在这里加，
                # 但因为 u0 已经是尽头，不影响后续逻辑，所以通常不需要写 cnt += 1

        return adj, s_mask
        
    def get_tc_dag_metadata(self, speakers, thread_nums, w_size):
        """
        实现 TC-DAG 特有的拓扑约束：
        1. 线索内序列连接 (Intra-Thread Sequence)
        2. 跨线索隔离 (Thread Isolation)
        3. 线索首句根锚定 (Root-Anchoring)
        """
        n = len(speakers)
        # adj[i, j] = 1 表示存在从 j 指向 i 的有向边
        adj = np.zeros((n, n), dtype=np.float32)
        # s_mask 用于区分关系类型 (同说话人=1, 异说话人=0)
        s_mask = np.zeros((n, n), dtype=np.int64)

        # 映射句子索引到其所属的线程索引及其在该线程内的相对位置
        thread_ends = list(accumulate(thread_nums))
        thread_starts = [w - z for w, z in zip(thread_ends, thread_nums)]
        
        sent2thread = {}
        sent_pos_in_thread = {}
        for t_idx, (start, end) in enumerate(zip(thread_starts, thread_ends)):
            for pos, s_idx in enumerate(range(start, end)):
                sent2thread[s_idx] = t_idx
                sent_pos_in_thread[s_idx] = pos

        for i in range(n):
            t_i = sent2thread[i]
            start_of_thread = thread_starts[t_i]
            
            # --- 规则 1 & 2: 线索内序列连接 & 线索间隔离 ---
            # 只在当前线索内部往前寻找前驱节点
            cnt = 0
            # 从当前句子的前一句开始，一直回溯到该线索的第一句
            for j in range(i - 1, start_of_thread - 1, -1):
                adj[i, j] = 1.0
                s_mask[i, j] = 1 if speakers[j] == speakers[i] else 0
                
                if speakers[j] == speakers[i]:
                    cnt += 1
                    # 遵循窗口大小 w 的限制
                    if cnt == w_size:
                        break

            # --- 规则 3: 线索首句根锚定 ---
            # 如果当前句子是该线索的起始句，且不是整个对话的第一句
            if i > 0 and i == start_of_thread:
                # 强制连接到对话的根话语 (u1, 索引为 0)
                adj[i, 0] = 1.0
                s_mask[i, 0] = 1 if speakers[i] == speakers[0] else 0
                    
        return adj, s_mask
    
    def get_dag_metadata(self, speakers, w_size):
        """
        仿照 DAG-ERC 的 get_adj_v1 逻辑
        speakers: 当前对话的说话人列表 [S0, S1, S2...]
        w_size: 往前寻找相同说话人的窗口大小 (config.windowp)
        """
        n = len(speakers)
        # adj[i, j] = 1 表示 j 是 i 的前驱
        adj = np.zeros((n, n), dtype=np.float32)
        # s_mask[i, j] = 1 表示 i, j 说话人相同；0 表示不同
        s_mask = np.zeros((n, n), dtype=np.int64)

        for i in range(n):
            cnt = 0 # 记录已找到的相同说话人数量
            # 严格 DAG：只往前找 (j < i)
            for j in range(i - 1, -1, -1):
                adj[i, j] = 1.0 # 建立有向边
                
                # 记录说话人关系
                if speakers[j] == speakers[i]:
                    s_mask[i, j] = 1
                    cnt += 1
                    # 如果达到了窗口限制 w，停止在该对话线索上的回溯
                    if cnt == w_size:
                        break
                else:
                    s_mask[i, j] = 0
                    
        return adj, s_mask
#数据预处理的终极装配线。它调用了之前我们分析过的所有“零件函数”（如 find_utterance_index, merge_same_thread, get_adj_matrix），将对话数据彻底转化为张量（Tensor）运算所需的索引。
# 读进来的还是文本字符串（"apple", "boy"），模型看不懂。所以预处理器紧接着把它们转成 ID。
    def transform2indices(self, dataset, mode='train'):
        res = []
        for document in dataset:
            sentences, speakers, replies, pieces2words = [document[w] for w in ['sentences', 'speakers', 'replies', 'pieces2words']]## 从数据字典中取出基础信息：句子内容、发言人、回复关系、分词映射表
            if 'train' in mode or 'valid' in mode:## 如果是训练或验证模式，还需要取出标注好的：四元组、目标、属性、观点
                triplets, targets, aspects, opinions = [document[w] for w in ['triplets', 'targets', 'aspects', 'opinions']]
            doc_id = document['doc_id']## 对话的唯一ID

            
            #计算位置索引（为了加上 [CLS] 和 [SEP]）
            # sentence_length = list(map(lambda x : len(x) + 2, sentences))
            sentence_length = list(map(lambda x : len(x) + 2, sentences))## sentence_length：计算每句话加入 [CLS] 和 [SEP] 后的总长度（即原句长 + 2）

            # token2sentid = [[i] * len(w) for i, w in enumerate(sentences)]# token2sentid：生成每个Token所属的句子ID。例如：[0,0,0, 1,1, 2,2,2]
            token2sentid = [[i] * len(w) for i, w in enumerate(sentences)]
            token2sentid = [w for line in token2sentid for w in line] # [0, 0, 0, 1, 1, 2, 2, 2, 2, 2] 分别代表句子的id

            token2speaker = [[11] + [w] * len(z) + [10] for w, z in zip(speakers, sentences)]# token2speaker：标记每个Token是谁说的。开头标11，结尾标10，中间是说话者ID
            token2speaker = [w for line in token2speaker for w in line] #[11, 0, 0, 1, 1, 2, 2, 2, 2, 10] 11代表开始，10代表结束，其它代表说话者的id

            # New token indices (with CLS and SEP) to old token indices (without CLS and SEP)# new2old：创建一个“新位置到旧位置”的映射表。# 因为加入了[CLS]和[SEP]，所有的词索引都往后挪了，这个字典用来找回原来的词。
            new2old = {}
            cur_len = 0
            for i in range(len(sentence_length)):
                for j in range(sentence_length[i]):
                    if j == 0 or j == sentence_length[i] - 1:
                        new2old[len(new2old)] = -1 
                    else:
                        new2old[len(new2old)] = cur_len
                        cur_len += 1

            tokens = [[self.config.cls] + w + [self.config.sep] for s, w in zip(speakers, sentences)]## tokens：给每句话的前后加上 [CLS] 和 [SEP]

            # sentence_ids of each token (new token)# nsentence_ids：记录新序列中每个 Token 属于第几句
            nsentence_ids = [[i] * len(w) for i, w in enumerate(tokens)]
            nsentence_ids = [w for line in nsentence_ids for w in line]

            flatten_tokens = [w for line in tokens for w in line]# flatten_tokens：把嵌套的句子列表压平，变成一维的长列表
            sentence_end = [i - 1 for i, w in enumerate(flatten_tokens) if w == self.config.sep]# 找到所有 [SEP] 和 [CLS] 在长列表中的物理位置
            sentence_start = [i + 1 for i, w in enumerate(flatten_tokens) if w == self.config.cls]
            # add speaker tokens at the end of each sentence# 把每句话结尾的 [SEP] 替换成说话人的 ID 词，增强模型对“谁在说话”的感知
            for ts, s in zip(tokens, speakers):
                ts[-1] = self.tokenizer.tokenize(str(s))[0]

            utterance_spans = list(zip(sentence_start, sentence_end))# utterance_spans：记录每句话真正的语义内容（扣除[CLS]和[SEP]）的起止区间
            utterance_index, token_index, thread_length, thread_nums, sent_idx2reply_idx = self.find_utterance_index(replies, sentence_length)# 调用之前的 find_utterance_index，计算线索长度、线索包含的句子数、回复映射表
            reply_mask, speaker_masks, thread_masks = self.get_neighbor(utterance_spans, replies, sum(sentence_length), speakers, thread_nums)# 调用 get_neighbor，生成回复掩码、发言人掩码、线索掩码（用于隔离不同线索的信息）
            
            # add reply adj# 构建“句子级”的回复邻接矩阵：如果句i回复了句j，则 adj[i][j]=1
            n = len(replies)
            dag_adj3, dag_s_mask3 = self.get_tc_dag_metadata2(speakers, thread_nums, getattr(self.config, 'windowp', 3))
           # print('=================')
          #  print('dag_adj_old')
          #  print(dag_adj2)
          #  print('-----------------')
          #  print('dag_adj_new')
          #  print(dag_adj3)
          #  print('=================')
            # DO speaker_adj# 构建“说话人级”的邻接矩阵：同一个人的话互联
            utterance_level_speaker_adj = np.array([[0.0] * n for i in range(n)])
            for i in range(len(speakers)):
                cur_speaker = speakers[i]
                for j in range(len(speakers)):
                    if cur_speaker == speakers[j]:
                        utterance_level_speaker_adj[i][j] = 1.0


            input_ids = list(map(self.tokenizer.convert_tokens_to_ids, tokens))# 将原始 Token ID 化

            input_masks = [[1] * len(w) for w in input_ids]
            input_segments = [[0] * len(w) for w in input_ids]

            if 'train' in mode or 'valid' in mode:# 修正 targets/aspects/opinions 的索引位置# 公式：新位置 = 原位置 + 2 * 句子ID + 1 (因为每句多了CLS和SEP)
                targets = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t in targets] # 2 * token2sentid[s] + 1 代表CLS+SEP
                aspects = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t in aspects]
                opinions = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t, p in opinions]
                opinions = list(set(opinions))

                full_triplets, new_triplets = [], []# 处理四元组（Triplets）并同步修正它们的起止索引
                # t_s-> target_start, t_e-> target_end, etc.
                for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in triplets:
                    new_index = lambda start, end : (-1, -1) if start == -1 else (start + 2 * token2sentid[start] + 1, end + 2 * token2sentid[start])
                    t_s, t_e = new_index(t_s, t_e)
                    a_s, a_e = new_index(a_s, a_e)
                    o_s, o_e = new_index(o_s, o_e)
                    line = (t_s, t_e, a_s, a_e, o_s, o_e, self.polarity_dict[polarity])
                    full_triplets.append(line)
                    if all(w != -1 for w in [t_s, a_s, o_s]):
                        new_triplets.append(line)   # with CLS+SEP
                # 1 relation # 将修正后的位置信息进行编码，生成实体列表、关系列表和极性列表
                relation_lists = self.wordpair.encode_relation(full_triplets) # eg (target head, aspect head, h2h)
                pairs = self.get_pair(full_triplets) # ta,to,ao full, eg (target_start, target_end, aspect_start, aspect_end)
                # 2 entity
                target_lists = self.wordpair.encode_entity(targets, 'ENT-T')
                aspect_lists = self.wordpair.encode_entity(aspects, 'ENT-A')
                opinion_lists = self.wordpair.encode_entity(opinions, 'ENT-O')
                entity_lists = target_lists + aspect_lists + opinion_lists     # eg. (target_start, target_end, 'ENT-T')
                # 3 polarity
                polarity_lists = self.wordpair.encode_polarity(new_triplets)  # eg（target head, opinion head, polarity)
            else:
                new_triplets, pairs, entity_lists, relation_lists, polarity_lists = [], [], [], [], []

            #DO merge_same_thread# 调用 merge_same_thread 将散装句子缝合成长线索，并定位非发言人内容的坐标
            merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length, nonspeaker_token_positions = \
            self.merge_same_thread(input_ids, input_masks, input_segments, sentence_length, utterance_index, thread_length)# thread_idxes：计算每个线程在长序列中的句子索引范围
            thread_range = [0]+list(accumulate(thread_nums))
            thread_range = [(thread_range[i], thread_range[i+1]) for i in range(len(thread_range)-1)]
            thread_idxes = [[i for i in range(start, end)] for (start, end) in thread_range][1:]
            thread_idxes = [[0] + w for w in thread_idxes]
            
            if 'dep' in mode:# 如果开启了语法依存（dep），则为每个 Thread 生成一个 $N \times N$ 的邻接矩阵
                piece_dep = document['piece_dep']
               
                if self.config.merged_thread == 0:
                    deprels, heads = [piece_dep[w] for w in ['deprels', 'heads']]
                    adj_matrixes = [self.get_adj_matrix(d, h) for d, h in zip(deprels, heads, )]
                    assert len(deprels) == len(heads)
                else:
                    thread_deprels, thread_heads,  = [piece_dep[w] for w in ['thread_deprels', 'thread_heads', ]]
                    adj_matrixes = [self.get_adj_matrix(d, h) for d, h in zip(thread_deprels, thread_heads, )]
                    assert  len(thread_deprels) == len(thread_heads)
                '''
                # === 插入打印代码 ===
                print(f"=== Doc ID: {doc_id} ===")
                print(f"句子总数 (外层长度): {len(input_ids)}")
                print(f"每句话的Token长度: {[len(sent) for sent in input_ids]}")
                print(f"=== Doc ID: {doc_id} 的 Merged 数据 ===")
                print(f"线索数量 (Thread Count): {len(merged_input_ids)}") 
                # 打印每个线索拼接后的总长度
                thread_lengths = [len(thread) for thread in merged_input_ids]
                print(f"每个线索的长度: {thread_lengths}")
                '''   
                res.append((doc_id, speakers, input_ids, input_masks, input_segments, sentence_length, nsentence_ids, utterance_index, token_index, 
                            thread_length, token2speaker, reply_mask, speaker_masks, thread_masks, pieces2words, new2old, 
                            new_triplets, pairs, entity_lists, relation_lists, polarity_lists, thread_idxes,
                            merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length, adj_matrixes, dag_adj3, dag_s_mask3, utterance_level_speaker_adj))
            
            else:
                res.append((doc_id, speakers, input_ids, input_masks, input_segments, sentence_length, nsentence_ids, utterance_index, token_index, 
                        thread_length, token2speaker, reply_mask, speaker_masks, thread_masks, pieces2words, new2old, 
                        new_triplets, pairs, entity_lists, relation_lists, polarity_lists, thread_idxes,
                        merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length))
        #print(len(res))
        return res
    

    def forward(self):
        # modes default: 'train valid test'
        modes = self.config.input_files
        datasets = {}

        for mode in modes.split():
            data = self.read_data(mode) # have been tokenized and aligned
            datasets[mode] = data

        label_dict = self.get_dict()

        res = {}
        for mode in modes.split():
            res[mode] = self.transform2indices(datasets[mode], mode)
        
        res['label_dict'] = label_dict
        return res