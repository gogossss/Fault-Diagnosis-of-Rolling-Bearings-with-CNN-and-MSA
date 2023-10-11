# 自学PYTHON
# 开发时间 2023/9/30 9:02
import numpy as np
import torch
from torch import nn, optim
from transformers import BertTokenizer, BertModel
from torchvision import transforms as T

class MyBertModel(nn.Module):
    def __init__(self, pretrain_path, max_length=512, cat_entity_rep=False, mask_entity=False):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity

    def forward(self, inputs):
        inputs = inputs
        outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
        return outputs.pooler_output

    def tokenize(self, raw_tokens):
        tokens = ['[CLS]']
        for token in raw_tokens:
            token = str(token).lower()
            tokens += self.tokenizer.tokenize(token)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask


class MY_CNN_LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_layers, output_size, batch_size):
        super(MY_CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=30, seq_len=24, input_size=7) ---> permute(0, 2, 1)
        # (30, 7, 24)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        # (batch_size=30, out_channels=32, seq_len-4=20) ---> permute(0, 2, 1)
        # (30, 20, 32)
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x['word'].float()
        t = x.size(0)
        x = x.view(self.batch_size, t, x.size(1))
        #TODO:用mask还能添加注意力机制
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.view(self.batch_size, t, -1)
        return x

    def tokenize(self, raw_tokens):
        # mask
        mask = np.zeros(self.hidden_size, dtype=np.int32)
        mask[:len(raw_tokens)] = 1

        return raw_tokens, mask


class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(my_sentence_encoder)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class Proto(FewShotREModel):

    def __init__(self, sentence_encoder, dot=False):
        FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(0.1)
        self.dot = dot

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, s, q, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.sentence_encoder(s)  # (B * N * K, D), where D is the hidden size
        query_emb = self.sentence_encoder(q)  # (B * total_Q, D)
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        hidden_size = support.size(-1)

        # mean_s = torch.mean(s['word'].float(), dim=0, keepdim=True)
        # std_s = torch.std(s['word'].float(), dim=0, keepdim=True)
        # support = (s['word'].float()-mean_s)/std_s
        # support = (support.float() - torch.min(support.float())) / (torch.max(support.float()) - torch.min(support.float()))

        # mean_q = torch.mean(q['word'].float(), dim=0, keepdim=True)
        # std_q = torch.std(q['word'].float(), dim=0, keepdim=True)
        # # query = (q['word'].float()-mean_q)/std_q
        # query = (support.float() - torch.min(support.float())) / (torch.max(support.float()) - torch.min(support.float()))

        support = support.view(-1, N, K, hidden_size)  # (B, N, K, D)
        query = query.view(-1, total_Q, hidden_size)  # (B, total_Q, D)

        # Prototypical Networks
        # Ignore NA policy
        support = torch.mean(support, 2)  # Calculate prototype for each class
        logits = self.__batch_dist__(support, query)  # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred



