import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,T)

        # mask out the future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # softmax the attention scores
        wei = F.softmax(wei, dim=-1) # (B,T,T)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B,T,head_size)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=32, block_size=8):
        super().__init__()
        self.block_size = block_size


        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # vocab_size:
        # It is the how many kinds of token we have
        # Because the generation is to find "Which token in the vocab possibly will be the next?"
        #  
        # n_embd:
        # How many length of vector do we use to describe a token?
        # Cause token is just a number, we can't find relations directory through two number 
        # So we use vector, let the high-diemsion features to decide the relations

        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.sa_head = Head(n_embd, n_embd, block_size)
        self.ffwd = FeedForward(n_embd=n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Base on the features(The embedding) we have, calculate the exact prossibility of next token

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb # (B,T,C)
        x = self.sa_head(x) # apply one head of self-attention (B,T,C)
        x = self.ffwd(x)
        logits = self.lm_head(x) # (B,T,Vocab_size)

        if targets is None:
            loss = None
        else:
            # To fit the format of F.cross_entropy
            B,T,C = logits.shape # Batch, Time(Token/Sequence Length), Channels
            logits = logits.view(B*T, C)
            targets = targets.view(-1)

            # Calculate loss
            loss = F.cross_entropy(logits, targets)
         
        return logits, loss
    
    def generate(self, idx:torch.Tensor, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # get predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step(The next word after the input)
            logits = logits[:, -1, :]
            # Convert Scores to Probabilities
            # That means Emvedding function don't generate possibility, only scores
            probs = F.softmax(logits, dim=-1)

            # We don't just pick the character with the highest probability
            # Instead, make a dice with different weights(Base on possibilities) on each face
            idx_next = torch.multinomial(probs, num_samples=1)
            # Connect the idx with new predictions idx_next
            idx = torch.cat((idx, idx_next), dim=1)
        return idx