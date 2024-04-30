import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import AlbertTokenizer, AlbertModel
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel

from torch.utils.data import DataLoader, Dataset


model_name = [
    "microsoft/deberta-v3-small", 
    "microsoft/deberta-v3-base", 
    "microsoft/deberta-v3-large",
    "deepset/roberta-base-squad2",
    "sentence-transformers/sentence-t5-base",
][-1]

class MarkdownDataset(Dataset):
    
    def __init__(self, df, max_len=256, mode='train'):
        super().__init__()
        #self.df = df

        self.max_len = max_len
        self.label = df['actual_embeddings'].values
        self.txt1 = df['original_text'].values
        self.txt2 = df['rewritten_text'].values

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                        #  do_lower_case=True
                                                      )
        self.mode=mode

    def __getitem__(self, index):
        #row = self.df.iloc[index]
        #label = row['actual_embeddings'][0]
        #txt1 = row['original_text']
        #txt2 = row['rewritten_text']
        label = self.label[index][0]
        txt1 = self.txt1[index]
        txt2 = self.txt2[index]
        
        inputs1 = self.tokenizer.encode_plus(
            txt1,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        ids1 = torch.LongTensor(inputs1['input_ids'])
        mask1 = torch.LongTensor(inputs1['attention_mask'])

        inputs2 = self.tokenizer.encode_plus(
            txt2,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )

        ids2 = torch.LongTensor(inputs2['input_ids'])
        mask2 = torch.LongTensor(inputs2['attention_mask'])

        

        return ids1, mask1, ids2, mask2, torch.FloatTensor(label)

    def __len__(self):
        return len(self.label)
