# Import necessary libraries for operating system operations, system-specific parameters, data processing, machine learning, and neural network functions
import os
import sys
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

# Import custom modules for dataset handling, model definitions, and adversarial training techniques
from dataset import MarkdownDataset
from model import M
from adv import FGM, EMA

# Get the fold number from command line arguments
fold = int(sys.argv[1])


# Function to adjust learning rate based on epoch number
def adjust_lr(optimizer, epoch, lr=5e-5):
    if epoch < 1:
        lr = lr
    elif epoch < 2:
        lr = lr
    elif epoch < 5:
        lr = lr
    else:
        lr = lr

    for p in optimizer.param_groups:
        p['lr'] = lr
        
    return lr
    
def get_optimizer(net, lr=5e-5):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, betas=(0.9, 0.999),
                                  weight_decay=1e-2,
                                  eps=1e-8 
                                 ) #1e-08)
    return optimizer


# Function to calculate a custom CV score using cosine similarity and raising to the third power
def CVScore(df):
    scs = lambda row: abs((cosine_similarity(row["actual_embeddings"], row["pred_embeddings"])) ** 3)
    

    df["actual_embeddings"] = df["label"].apply(get_embedding) #.progress_apply(get_embedding)
    df["pred_embeddings"] = df["pred"].apply(get_embedding )
    
    a = np.vstack(df["actual_embeddings"])
    b = np.vstack(df["pred_embeddings"])
    score = ((a * b).sum(1) ** 3).mean()
    return score


# Load embedding data and prompts for embedding lookup
matrix = np.load('../data/matrix.npy')
prompts = joblib.load('../data/prompts.joblib')
dict_embedding = {prompts[i]: matrix[i][None] for i in range(len(prompts))}


# Function to fetch or compute embeddings using a model
def get_embedding(text):
    if text not in dict_embedding:
        dict_embedding[text] = model_t5.encode(text, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)
    return dict_embedding[text]


tqdm.pandas()

# Load training and testing datasets
data_train = pd.read_parquet('../data/data_train.parquet')
data_test = pd.read_parquet('../data/data_test.parquet')
tot = len(data_train)
indexes = [i for i in range(tot) if i % 8 != fold]
data_train = data_train.iloc[indexes]

# Apply embeddings
data_train["actual_embeddings"] = data_train["label"].apply(get_embedding)
data_test["actual_embeddings"] = data_test["label"].apply(get_embedding)

# Constants for data loading and model training
BS = 32  # Batch size
NW = 8  # Number of workers
MAX_LEN = 256  # Maximum length of input sequences
train_ds = MarkdownDataset(data_train, max_len=MAX_LEN, mode='train')
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=NW, pin_memory=True, drop_last=True)
test_ds = MarkdownDataset(data_test, max_len=MAX_LEN, mode='train')
test_loader = DataLoader(test_ds, batch_size=BS * 4, shuffle=False, num_workers=NW, pin_memory=True, drop_last=False)


# Function to handle data fetching for the model
def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


# Function to evaluate the model on the test set
def eval_model():
    model.eval()
    pred_list = []
    for item in tqdm(test_loader):
        inputs, target = read_data(item)
        with torch.no_grad():
            pred = model(*inputs)
        pred = pred.detach().cpu().numpy()
        score = np.einsum('nd,md->nm', pred, matrix)
        indexs = score.argmax(1)
        pred_prompt = [prompts[item] for item in indexs]
        pred_list.extend(pred_prompt)
    data_test['pred'] = pred_list
    score = CVScore(data_test)
    print(f"CV Score: {score}")

    return score 


# Training setup
epochs = 3
learning_rate = 5e-5
iters_to_accumulate = 1
model = M().cuda()
ema = EMA(model, 0.999)
ema.register()
optimizer = get_optimizer(model, lr=learning_rate)
num_train_steps = int(len(data_train) / (BS) * epochs)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

mixed_precision = False

def criterion(x, y):
    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    y = torch.nn.functional.normalize(y, p=2, dim=-1)
    corr = ( (x * y).sum(-1) ** 3 ).mean()
    return -corr

def criterion_simcse(x, y):
    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    y = torch.nn.functional.normalize(y, p=2, dim=-1)

    bs = x.shape[0]
    mask = torch.eye(bs).float()
    mask = mask.to(x)

    corr_matrix = torch.einsum('ai,bi->ab', x, y) # ** 3

    labels = torch.arange(bs).to(x).long()
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(corr_matrix * 10, labels)


    return loss

def metric_(x, y):
    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    y = torch.nn.functional.normalize(y, p=2, dim=-1)
    corr = ( (x * y).sum(-1) ** 3 ).mean()
    return -corr



# Gradient scaling for mixed precision training
scaler = GradScaler()

# Main training loop
for e in range(epochs):
    model.train()
    tbar = tqdm(train_loader, file=sys.stdout)
    loss_list, metric_list, preds, labels = [], [], [], []
    counter = 0
    for idx, data in enumerate(tbar):
        counter += 1
        inputs, target = read_data(data)
        with autocast(device_type="cuda", dtype=torch.float32):
            pred = model(*inputs)
            loss = criterion(pred, target) / iters_to_accumulate
        scaler.scale(loss).backward()
        if (counter + 1) % iters_to_accumulate == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        ema.update()
        scheduler.step()
        loss_list.append(loss.detach().cpu().item())
        metric_list.append(metric_(pred, target).detach().cpu().item())
        avg_loss = np.round(np.mean(loss_list), 4)
        avg_metric = np.round(np.mean(metric_list), 4)
        tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {learning_rate} score: {avg_metric}")

ema.apply_shadow()
score = eval_model()
output_model_file = f"../output/model_output2_{fold}/{score}.bin"
model_to_save = model.module if hasattr(model, 'module') else model
torch.save(model_to_save.state_dict(), output_model_file)
ema.restore()

pd.DataFrame(prompts).to_csv(f'../output/model_output2_{fold}/submit_prompt.csv', index=None)
