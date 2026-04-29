import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        # 1. Add special tokens using enumerate
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        
        self.vocab_size = len(special_tokens)
        
        # 2. Collect unique lowercase words
        unique_words = set()
        for text in texts:
            unique_words.update(text.lower().split())
        
        # 3. Sort words
        sorted_words = sorted(unique_words)
        
        # 4. Add words to vocab
        for word in sorted_words:
            self.word_to_id[word] = self.vocab_size
            self.id_to_word[self.vocab_size] = word
            self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        words = text.lower().split()
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(word, unk_id) for word in words]
    
    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            word = self.id_to_word.get(i, self.unk_token)
            if word in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            words.append(word)
        return " ".join(words)