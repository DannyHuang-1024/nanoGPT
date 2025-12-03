import torch
from tokenizer import Tokenizer
from model import BigramLanguageModel

# -----------------------------------------------------------------------------
# CONFIGURATION (MUST MATCH YOUR TRAIN.PY SETTINGS EXACTLY)
# -----------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "gpt_model.pth"
TOKENIZER_PATH = "resources/TokenizerModel.json"

# These must match what you used in train.py!
N_EMBD = 384
BLOCK_SIZE = 256
N_LAYER = 6  # Or whatever you set in BigramLanguageModel default or passed in
N_HEAD = 4   # This is currently hardcoded in your model.py Block class

def load_model():
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.load_json(TOKENIZER_PATH)
    vocab_size = len(tokenizer.vocab)

    print("Initializing model architecture...")
    # Initialize the model structure (with random weights first)
    model = BigramLanguageModel(vocab_size=vocab_size, 
                                n_embd=N_EMBD, 
                                block_size=BLOCK_SIZE, 
                                n_layer=N_LAYER)

    print(f"Loading weights from {MODEL_PATH}...")
    # Load the actual trained weights
    # map_location ensures it loads on CPU if you don't have a GPU right now
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    
    # IMPORTANT: Switch to eval mode!
    # This turns off Dropout layers so generation is deterministic and clean
    model.eval() 
    
    return model, tokenizer

if __name__ == "__main__":
    # 1. Load resources
    model, tokenizer = load_model()

    # 2. Prepare the input prompt
    start_text = "The quick brown fox"
    print(f"\nPrompt: {start_text}")
    print("-" * 50)

    # Encode
    input_ids = tokenizer.encode(start_text)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    # 3. Generate
    # We use torch.no_grad() because we don't need to calculate gradients for generation
    # It saves memory and makes it faster
    with torch.no_grad():
        generated_ids = model.generate(input_tensor, max_new_tokens=500)

    # 4. Decode and Print
    # [0] is because generate returns (Batch, Time), we want the first batch item
    output_text = tokenizer.decode(generated_ids[0].tolist())
    
    print(output_text)