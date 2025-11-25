import torch

from tokenizer import Tokenizer

with open("./resources/harrypotter.txt", 'r') as f:
    text = f.read()

tk = Tokenizer()

print(tk.encode(text[:1000]))