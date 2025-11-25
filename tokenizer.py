import json
import base64
import regex as re

import argparse
import os

from tqdm import tqdm

class Tokenizer:
    def __init__(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

        pass

    def train(self, text, vocab_size, verbose=False):
        if not text:
            print("Warning: training text is empty.")
            return [], self.vocab
        # tokens = text.encode('utf-8')
        chunks = self.split(text)
        # tokens = list(tokens) # Without using map(int, ) here, because list() has the function
        
        ids = [list(word.encode('utf-8')) for word in chunks]
    
        
        if(vocab_size > 256):
            num_merges = vocab_size - 256
            
            # Using 'with' allows us to update the bar with stats
            with tqdm(total=num_merges, desc="Training BPE") as pbar:
                for i in range(num_merges):
                    stats = self.get_stats(ids)
                    current_max_pair = max(stats, key=stats.get)
                    occurrence_count = stats[current_max_pair]
                    
                    idx = 256 + i
                    ids = self.merge_ids_list(ids, current_max_pair, idx)
                    self.merges[current_max_pair] = idx

                    p0, p1 = current_max_pair[0], current_max_pair[1] 
                    self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
                    
                    # Update progress bar description with the pair and count
                    pbar.set_postfix({"Merging": current_max_pair, "Count": occurrence_count})
                    pbar.update(1)

        print("Training finished! Trained by "
              f"text with length:{len(chunks)}")
        print(f"Vocab length:{len(self.vocab)}")
        
        
        return ids, self.vocab
    
    def split(self, text):
        pattern = re.compile(self.GPT4_SPLIT_PATTERN)
        return pattern.findall(text) # -> list[Any]:

        
    
    def save(self, file_path = r'./model.json'):
        '''
        Saves the merges and vocab dictionaries to local storage.
        The model saved as a JSON file.
        '''

        # This step convert tuple keys to strings 
        merges_str_keys = {str(k): v for k, v in self.merges.items()}

        # This step convert Byte value in vocab to strings
        vocab_b64_value = {k: base64.b64encode(v).decode('utf-8') 
                           for k, v in self.vocab.items()}

        model_data = {
            "merges": merges_str_keys,
            "vocab" : vocab_b64_value
        }

        with open(file_path, 'w') as f:
            json.dump(model_data, f, indent=2)
            # indent specifies the number of spaces to use for indentation
        print(f"Tokenizer saved to {file_path}")
    
    @classmethod
    def load_json(cls,file_path):
        '''
        Loads a tokenizer from a JSON file.
        '''
        tokenizer = cls()
        with open(file_path, 'r') as f:
            model_data = json.load(f)

        # Convert strings keys back to tuples for merges
        merges_tuple_keys = {eval(k): v for k,v in model_data["merges"].items()}

        # Convert base64 strings back to bytes for vocab
        vocab_Byte_values = {int(k) : base64.b64decode(v) for k,v in model_data["vocab"].items()}

        tokenizer.merges = merges_tuple_keys
        tokenizer.vocab = vocab_Byte_values
        print(f"Tokenizer loaded from {file_path}")
        return tokenizer

        

    def encode(self, text):
        '''
        Use the pre-trained model dictionary to encode text
        text : string 
        '''

        chunks = self.split(text)
        final_ids = []

        for chunk in chunks:
            tokens = list(chunk.encode("utf-8"))

            while len(tokens) >= 2:
                stats = self.get_stats(tokens)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                tokens = self.merge(tokens, pair, idx)
            final_ids.extend(tokens)
        return final_ids

    def decode(self, ids):
        '''
        ids : [int, int, ...]
        '''
        vocab = self.vocab
        tokens = b"".join(vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")
        

    def get_stats(self, ids):
        '''
        input a list of number tokens, count the occur times of every pair 
        ids : [int, int, ...]
            or [[int, int, ...], [int, int, ...], ...]
        return : {(pair) : times}  
        '''
        stats = {}
        if type(ids[0]) == list:
            for word in ids:
                # stats = stats | self.get_stats(word)
                # This line above will choose the count of the latter
                # when duplicate values occur
                word_stats = self.get_stats(word)
                for pair, count in word_stats.items():
                    stats[pair] = stats.get(pair, 0) + count
                # Concatenate dictionaries into one
            return stats

        pairs = zip(ids, ids[1:])
        
        for pair in pairs:
            stats[pair] = stats.get(pair, 0)+1

        return stats

    def merge_ids_list(self, ids_list, pair, idx):
        '''
        Update: Applies the merge to each sublist in a list
        '''
        new_ids_list = []
        
        for ids in ids_list:
            newids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                    newids.append(idx)
                    i += 2
                else:
                    newids.append(ids[i])
                    i += 1
            new_ids_list.append(newids)
        return new_ids_list
    
    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--save_dir", default="./")
    parse.add_argument("--sources_dir", default="./")
    parse.add_argument("--vocab_num", default=400)


    regexToken = Tokenizer()

    args = parse.parse_args()

    all_text_chunks = []

    print("Loading files......")
    for root, dirs, files in os.walk(args.sources_dir):
        for filename in files:
            full_path = os.path.join(root, filename)
            
            if(filename.endswith('.txt')):
                with open(f'{full_path}', 'r', encoding='utf-8') as f:
                    all_text_chunks.append(f.read())
    # Join all chunks into one massive string
    combined_text = "\n\n".join(all_text_chunks)
    print(f"Loaded {len(all_text_chunks)} files. Total characters: {len(combined_text)}")
    
    # This ensures global statistics and unique IDs
    if combined_text:
        regexToken.train(combined_text, int(args.vocab_num))
        regexToken.save(os.path.join(args.save_dir, 'TokenizerModel.json'))
    else:
        print("No text found to train on.")

# python tokenizer.py --save_dir ./resources --sources_dir ./resources/TokenizerTrain/ --vocab_num 2000

