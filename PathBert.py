import torch.nn as nn
import os
import pickle
import numpy as np

from transformers import BertModel, BertTokenizer, BertConfig
import torch
# 参考链接：https://blog.csdn.net/qq_33812659/article/details/107437373
# 导入BertTokenizer和BertModel 该路径和模型存放的路径相同

class PathBert(nn.Module):
    def __init__(self, d_relation2id, d_entity2id):
        super(PathBert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('../../Models/bert_base_uncased')
        self.model = BertModel.from_pretrained('../../Models/bert_base_uncased').to("cuda") # 记得把model的device设置为cuda
        # '../../Models/bert_base_uncased'
        self.d_relation2id = d_relation2id
        self.d_entity2id = d_entity2id


    def getPathFromTensor(self, path_trace):
        last_r, e_t = path_trace[-1]
        # 得到key为id value为entity的字典
        d_entity2id = self.d_entity2id
        d_id2entity = {value: key for key, value in d_entity2id.items()}
        d_relation2id = self.d_relation2id
        d_id2relation = {value: key for key, value in d_relation2id.items()}
        # print(last_r,e_t)
        last_r_array = last_r.tolist()
        e_t_array = e_t.tolist()

        path_real_value = ''
        for last_r_index, e_t_index in zip(last_r_array, e_t_array):
            relation_temp = d_id2relation[last_r_index]
            entity_temp = d_id2entity[e_t_index]
            path_real_value = path_real_value + relation_temp + ' ' + entity_temp + ' '
        return path_real_value

    def getEmbedding(self, sentence):
        # 修改输入格式
        # 获得三个向量，input_ids;token_type_ids;attention_mask,三个响亮存储在一个字典中
        text_dict = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_attention_mask=True)

        # 模型对三个输入向量的维度要求为(batch, seq_len),所以要对text_dict重的三个向量进行升维
        input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0).to("cuda")
        token_type_ids = torch.tensor(text_dict['token_type_ids']).unsqueeze(0).to("cuda")
        attention_mask = torch.tensor(text_dict['attention_mask']).unsqueeze(0).to("cuda")
        # res是一个包含2个元素的turple （根据config的不同，返回的元素个数也不同）一个是sequnce_output（对应下面的last_hidden_state），一个是pooler_output。
        res = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 拿到该句话的每个字的embedding
        embedding = res[0].detach().squeeze(0)
        return embedding


    def forward(self, inputs):
        path_true_value = self.getPathFromTensor(inputs) # 得到真实的path值
        embedding = self.getEmbedding(path_true_value) # 得到path对应的嵌入

        return embedding # TODO: