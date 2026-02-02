import torch

import torch.nn as nn
from torch.nn import functional as F

#hyperparameters

batch_size = 62 # how many independent process will be carried out in parallel
block_size = 256 # what is the maximum context length for the length f predictions 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 280
n_embd = 384 # number of emmbeded dimensions
n_head = 6 # number of heads in multi head attention 
n_layers = 6 # number of transformer blocks
dropout = 0.2
#-----------------------------------------------

torch.manual_seed(1337)

#wget....
with open('input_clean.txt', 'r', encoding= 'utf-8') as f:
    text = f.read()
# here are all the unique charecters that occur in the whole text

chars = sorted(list(set(text)))
vocab_size = len(chars)

#creating a mapping from charecters to integers to tokenize and all
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i, ch in  enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]




def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)- block_size,(batch_size,))
    
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1: i + block_size +1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y




@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k]= loss.item()
        out[split]= losses.mean()
    model.train()
    return out






class MultiheadAttention(nn.Module):
    # multiple heads of self attention in parallel so that the model can jointly attend to information from different representation subspaces at different positions
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # creating multiple heads how many heads we want
        self.proj = nn.Linear(n_embd, n_embd) # final projection layer to bring back the concatenated output to original embedding dimension
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # taking all the heads and concatenating them along the channel dimension
        out = self.dropout(self.proj(out)) # projecting it back to original embedding dimension
        return out






class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__() # each token directly reads off logits for the next token from a lookup table this means that each token will have its own embedding
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias =False)
        self.value = nn.Linear(n_embd, head_size, bias =False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,_ = x.shape # batch size, time steps, channels
        k = self.key(x)   #(B,T,head_size)
        q = self.query(x) #(B,T,head_size)  
        #compute attention scores
        wei = wei = q @ k.transpose(-2,-1) * (k.shape[-1] ** -0.5) #(B,T,T) from Btc x Bct --> Btt this is like calculating the similarity scores between query and key
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf')) # this is to make sure that the model does not communicate with the future tokens, a decoder block
        wei = F.softmax(wei, dim=-1) #(B,T,T) attention (q,k,v) = softmax(q@k.T/sqrt(d_k)) @ v where q is query k is key v is value and d_k is the dimension of key
        wei = self.dropout(wei)
        v = self.value(x) #(B,T,head_size) so now we have the value matrix which we will be using to get the weighted sum
        out = wei @ v #(B,T,head_size)  Btt x Btc --> Btc
        return out





class FeedForward(nn.Module): # this is a simple feed forward neural network that allows the model to process the information after self attention by transforming it 
    # through non linear layers how this basically works is that each position is processed independently and identically and then the output is passed to the next layerq
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( 
            nn.Linear(n_embd,4* n_embd),  # first linear layer, multiplying by 4 as per the transformer paper because it helps in better representation
            # the above basically increases the size of the side of the residual block expands it and then in the next Linear layer we bring it back to original size
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),  # second linear layer why second linear layer because we want to bring it back to original dimension 
            nn.Dropout(dropout),
        ) 
    # this is a simple 2 layer feed forward neural network with ReLU activation in between which takes in n_embd dimensions and outputs n_embd dimensions
    def forward(self, x):
        return self.net(x)
    





class Block(nn.Module):
    # transformer block : communication followed by computation
    def __init__(self, n_embd, n_head):
        # n_head : number of heads we want in the multi head attention
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # layer norm normalizes the inputs across the features dimension, it basically normalizes each embedding vector into mean 0 variance 1 and then scales and shifts it via learnable parameters
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connection for self attention
        x = x + self.ffwd(self.ln2(x)) # residual connection for feed forward network
        return x


# super simple Bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token directly reads off logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # n_embd  --> number of variables embeded
        self.positional_embedding_table = nn.Embedding(block_size, n_embd) # each position gets its own embedding
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layers)]) # stack of transformer blocks
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # language model head to project the output to vocab size

    def forward(self, idx, targets = None):
        B,T = idx.shape
        token_emb = self.token_embedding_table(idx) # (B,T,C) B--> batch size T--> time steps C--> channels (n_embd)
        pos_emb = self.positional_embedding_table(torch.arange(T, device = device)) #(T,C), this is broadcasted along the batch dimension
        x = token_emb + pos_emb #(B,T,C), x holds not just token identiry info but also their position info
        x = self.blocks(x) #(B,T,C) passing through the transformer blocks 
        x = self.ln_f(x) #(B,T,C) final layer norm
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets= targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
            
        return logits,loss


    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  #crop to the last block size tokens
            #getting the preddiction 
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:, -1,:]    
            # apply softmax to get the probablities
            probs = F.softmax(logits, dim=-1) #(B,C) dim -1 matlab it automatically subtracts on of the dim or so i think need clarity
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,T)
            # append sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)
        return idx
        
    

model = BigramLanguageModel()
m = model.to(device)

# creating a pytorch optimizer

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train los {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    #sample a batch of data 
    xb, yb = get_batch('train')
    
    
    #evaluate the loss
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
#generate from the model

context = torch.zeros((1,1), dtype=torch.long, device = device)
print(decode(m.generate(context, max_new_tokens = 1000)[0].tolist()))    







