import sys, os
from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel

import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import AlbertTokenizer, AlbertModel

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch import optim


MAX_LEN = 256

model_name = [
    "microsoft/deberta-v3-small", 
    "microsoft/deberta-v3-base", 
    "microsoft/deberta-v3-large",
    "deepset/roberta-base-squad2",
    "sentence-transformers/sentence-t5-base",
][-1]
    
class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        if model_name in ['sentence-transformers/sentence-t5-base', 'sentence-transformers/sentence-t5-large',]:
            self.bert = AutoModel.from_pretrained(model_name).encoder
        else:
            self.bert = AutoModel.from_pretrained(model_name)
        
        self.mlp1 = nn.Linear(self.bert.config.hidden_size, 768)
        self.dropout = nn.Dropout(0.3)
                
    def forward(self, ids1, mask1, ids2, mask2):

        ids = torch.cat((ids1[:,:-1], ids2[:,:]), 1)
        mask = torch.cat((mask1[:,:-1], mask2[:,:]), 1)
        
        x0 = x = self.bert(ids, mask)[0]
        
        x = x.mean(1)/mask.float().mean(-1)[:,None].clip(1e-6)
        
        x = self.mlp1(self.dropout(x))
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        
        return x
