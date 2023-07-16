import torch.nn as nn
import os
import pickle
import numpy as np

from transformers import BertModel, BertTokenizer, BertConfig
import torch
# 参考链接：https://blog.csdn.net/qq_33812659/article/details/107437373
# 导入BertTokenizer和BertModel 该路径和模型存放的路径相同

class QuestionBert(nn.Module):
    def __init__(self, d_word2id):
        super(QuestionBert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('../../Models/bert_base_uncased')
        self.model = BertModel.from_pretrained('../../Models/bert_base_uncased').to("cuda") # 记得把model的device设置为cuda
        # '../../Models/bert_base_uncased'
        self.d_word2id = d_word2id

    def getQuestionsFromTensor(self, batch_question):
        questionArrays = batch_question.tolist()
        # 得到key为id value为word的字典
        d_word2id = self.d_word2id
        d_id2word = {value: key for key, value in d_word2id.items()}
        questions = []
        for questionArray in questionArrays:
            question = ''
            # [1,2,3,4,5,6]
            for index in questionArray:
                word = d_id2word[index]
                question = question + ' ' + word
            question = question.strip()
            questions.append(question)
        return questions
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
    def getEmbeddingFromQuestions(self, questions):
        embeddings = []
        for question in questions:  # 对question进行遍历
            temp_embedding = self.getEmbedding(question) # 得到单个问题的embedding
            embeddings.append(temp_embedding)  # 将单个问题的embedding嵌入embeddings数组中
        return embeddings  # 返回embedding



    def forward(self, inputs):
        questions = self.getQuestionsFromTensor(inputs) #
        embeddings = self.getEmbeddingFromQuestions(questions)  # TODO： tensor的list 报错了！！
        # 将tensor的list拼接成完整的tensor
        # 找到最大句子长度
        max_length = max([embedding.shape[0] for embedding in embeddings])
        # 创建一个用于合并的空白Tensor，形状为(batch_size, max_length, 768) 初始值为0
        combined_embedding_tensor = torch.zeros((len(questions), max_length, 768)).to("cuda")
        for i, embedding in enumerate(embeddings):
            length = embedding.shape[0]
            combined_embedding_tensor[i, :length, :] = embedding
        # embeddings = self.embedding(inputs)
        # embeddings = self.EDropout(embeddings)
        return combined_embedding_tensor # TODO: