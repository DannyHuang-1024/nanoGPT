import torch
from tqdm import tqdm

from tokenizer import Tokenizer
from model import BiagramLanguageModel

# -----------------------------------------------------------
# Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8
EPOCHS = 30000
EVAL_INTERVAL = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
# -----------------------------------------------------------


with open("./resources/harrypotter.txt", 'r') as f:
    text = f.read()


token = Tokenizer.load_json(r"resources/TokenizerModel.json")

# Turn our text into tensor
data = torch.tensor(token.encode(text), dtype=torch.long)
# print(data.shape, data.dtype)

# Split up the data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1024)
vocab_size = len(token.vocab)

def get_batch(data:torch.Tensor, batch_size, block_size):
    # Generate random start indexs from 0 to len(data) - block_size 
    # Here minus block_size is to avoid going out of bound
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

blm = BiagramLanguageModel(vocab_size=vocab_size).to(device)
optimizer = torch.optim.AdamW(blm.parameters(), lr = 1e-3)

def train(n_epochs, batch_size, block_size, 
          model:torch.nn.Module,
          optimizer:torch.optim.Optimizer):
    for step in tqdm(range(n_epochs), desc="Training progress"):
        # sample a batch of data
        xb, yb = get_batch(train_data, batch_size=4, block_size=8)
        xb, yb = xb.to(device), yb.to(device) 

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())

train(EPOCHS,
      batch_size=BATCH_SIZE,
      block_size=BLOCK_SIZE,
      model=blm,
      optimizer=optimizer
      )

sentences = "Harry"
inputs = torch.tensor(token.encode(sentences)).unsqueeze(0).to(device)
print(token.decode((blm.generate(inputs, max_new_tokens=100)[0].tolist())))