import torch
import torch.nn as nn
from torch.nn import functional as F

class BiagramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # To fit the format of F.cross_entropy
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)

            # Calculate loss
            loss = F.cross_entropy(logits, targets)
         
        return logits, loss
    
    def generate(self, idx:torch.Tensor, max_new_tokens):
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
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