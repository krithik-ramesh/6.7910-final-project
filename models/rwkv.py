import math, os
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

class ChannelMix(nn.Module):
    def __init__(self, layer_id, n_layer, n_embed):
        super().__init__()
        self.layer_id = layer_id
        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - layer_id/n_layer
            x = torch.ones(1,1, n_embed)
            for i in range(n_embed):
                x[0, 0, i] = i/n_embed
            
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        
        hidden_size = 4*n_embed
        self.key = nn.Linear(n_embed, hidden_size, bias=False)
        self.receptance = nn.Linear(n_embed, n_embed, bias=False)
        
        self.value = nn.Linear(hidden_size, n_embed, bias=False)
        
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + (1-self.time_mix_k) * xx

        xr = x * self.time_mix_r + (1-self.time_mix_r) * xx
        
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        
        kv = self.value(k)
        
        
        
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        
        return rkv
    

class TimeMix(nn.Module):
    def __init__(self, layer_id, n_layer, n_embed):
        super().__init__()
        self.layer_id = layer_id
        
        attn_sz = n_embed
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - layer_id/n_layer
            ratio_0_to_1 = layer_id / (n_layer - 1)
            
            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                
            self.time_decay = nn.Parameter(decay_speed)
            
            zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(attn_sz)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)
                
            
            x = torch.ones(1,1, n_embed)
            for i in range(n_embed):
                x[0, 0, i] = i/n_embed
            
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) +0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        
            self.aa = nn.Parameter(torch.ones(1,1,attn_sz))
            self.bb = nn.Parameter(torch.ones(1,1,attn_sz))
            pp = torch.ones(1,1,attn_sz)
            pp = pp * -1e30
            self.pp = nn.Parameter(pp)
            self.xx = nn.Parameter(torch.ones(1,1,attn_sz))
        
        hidden_size = attn_sz
        self.key = nn.Linear(n_embed, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embed, attn_sz, bias=False)
        
        self.value = nn.Linear(hidden_size, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, n_embed, bias=False)
        
    def forward(self, x):
        
        xx = self.xx

       
        xk = x * self.time_mix_k + (1-self.time_mix_k) * xx
        xv = x * self.time_mix_v + (1-self.time_mix_v) * xx
        xr = x * self.time_mix_r + (1-self.time_mix_r) * xx
        
        k = self.key(xk)
        
        
        v = self.value(xv)
        r= self.receptance(xr)
        
            
        r =torch.sigmoid(r)
        
        # Calculate the difference in size along the non-singleton dimension
        diff = k.shape[1] - self.aa.shape[1]
    
        
        b,t,c = x.shape
        aa = torch.nn.functional.pad(self.aa, (0, 0, 0, diff, 0, 0))
        bb = torch.nn.functional.pad(self.bb, (0, 0, 0, diff, 0, 0))
        pp = torch.nn.functional.pad(self.pp, (0, 0, 0, diff, 0, 0))
             
            
        ww = self.time_first + k
    
        
        qq = torch.maximum(pp, ww )
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        
        a = e1 * aa + e2 * v
        
        b = e1 * bb + e2
        wkv = a / b
        
        ww = pp + self.time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        
        with torch.no_grad():
            xx = nn.Parameter(x)
            self.aa = nn.Parameter(e1 * aa + e2 * v)
            self.bb = nn.Parameter(e1 * bb + e2)
            self.pp = nn.Parameter(qq)
            
        
        return self.output(r * wkv)
    
class Block(nn.Module):
    def __init__(self, layer_id, n_layer, n_embd):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        if self.layer_id == 0 :
            self.ffnPre = ChannelMix(0, n_layer, n_embd)
        else:
            self.att = TimeMix(layer_id, n_layer, n_embd)

        self.ffn = ChannelMix(layer_id, n_layer, n_embd)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)        
        if self.layer_id == 0 :
            x = x + self.ffnPre(self.ln1(x))  # better in some cases
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class RWKV(nn.Module):
    def __init__(self, n_layer, vocab_size,  n_embd, ctx_len):
        super().__init__()
        self.step = 0
        self.ctx_len = ctx_len
        self.emb = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.Sequential(*[Block(i, n_layer, n_embd)
                                    for i in range(n_layer)])

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
            idx = idx.to(self.emb.weight.device)

            self.step += 1
            
            B, T = idx.size()
            assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

            x = self.emb(idx)
            x = self.blocks(x)
            x = self.ln_out(x)

            x = self.head(x)
            

            loss = None
            if targets is not None:
                loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.to(x.device).view(-1))
            x = torch.mean(x, dim=0, keepdim=True)
            return x, loss
        
    def generate(self, idx, max_new_tokes):
        for _ in range(max_new_tokes):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx

with open('wiki_6500.txt', 'r', encoding='utf-8') as f:
    text = f.read()
len(text)
chars = sorted(list(set(text)))
vocab_size = len(chars)
block_size = 32
stoi = { ch:i for i, ch in enumerate(chars) }
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[x] for x in l])
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
train_data[:block_size+1]
batch_size = 4


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

import matplotlib.pyplot as plt

def visualize_attention_maps(model, idx):
    attn_weights, loss = model(idx)
    attn_weights = attn_weights.detach().cpu().numpy()

    # Plot attention maps
    for i, attn_map in enumerate(attn_weights):
        plt.figure(figsize=(10, 10))
        plt.imshow(attn_map)
        plt.title(f'Attention Map {i+1}')
        plt.colorbar()
        plt.show()

# Usage
model = RWKV(vocab_size=vocab_size, n_embd=vocab_size, ctx_len=32,n_layer=1)
idx, _ = get_batch('train')
visualize_attention_maps(model, idx)