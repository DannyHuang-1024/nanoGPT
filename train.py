import torch
from tqdm import tqdm

from tokenizer import Tokenizer
from model import BigramLanguageModel

# -----------------------------------------------------------
# Hyperparameters
# MAX_TRAIN_SIZE = 10 * 1024 * 1024
BATCH_SIZE = 64
BLOCK_SIZE = 256
EPOCHS = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_layer = 6
# -----------------------------------------------------------

def get_batch(data:torch.Tensor, batch_size, block_size):
    # Generate random start indexs from 0 to len(data) - block_size 
    # Here minus block_size is to avoid going out of bound
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model:torch.nn.Module):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(train_data if split == 'train' else val_data, 
                               batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(n_epochs, batch_size, block_size, 
          model:torch.nn.Module,
          optimizer:torch.optim.Optimizer):
    # for step in tqdm(range(n_epochs), desc="Training progress"):
    pbar = tqdm(range(n_epochs), desc="Training progress")
    for step in pbar:
        # sample a batch of data
        xb, yb = get_batch(train_data, batch_size=batch_size, block_size=block_size)
        xb, yb = xb.to(device), yb.to(device) 

        if step % eval_interval == 0 or step == n_epochs - 1:
            losses = estimate_loss(model)
            # Update the info shown RIGHT NEXT to the progress bar
            # pbar.set_postfix({
            #     "Train Loss": f"{losses['train']:.4f}",
            #     "Val Loss": f"{losses['val']:.4f}"
            # })
            
            # We use tqdm.write instead of print() to avoid messing up the bar
            tqdm.write(f"Step {step}: Train {losses['train']:.4f}, Val {losses['val']:.4f}")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())


if __name__ == "__main__":
    # with open("/media/danny/SuperMoose/Data/nanoGPT/pile_clean.txt", 'r') as f:
    #     text = f.read(MAX_TRAIN_SIZE)
    with open(r'./resources/input.txt', 'r') as f:
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

    blm = BigramLanguageModel(vocab_size=vocab_size, 
                              n_embd=n_embd, 
                              block_size=BLOCK_SIZE,
                              n_layer=n_layer).to(device)
    optimizer = torch.optim.AdamW(blm.parameters(), lr = 1e-3)

    train(EPOCHS,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        model=blm,
        optimizer=optimizer)

    sentences = "First Citizen:"
    inputs = torch.tensor(token.encode(sentences)).unsqueeze(0).to(device)
    print(token.decode((blm.generate(inputs, max_new_tokens=100)[0].tolist())))