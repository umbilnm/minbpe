from typing import List
from .base import Tokenizer, count_pairs_of_tokens, merge


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text: str, vocab_size: int):
        assert vocab_size >= 256, "Vocab size cant be less than 256"
        merges = {}
        vocab = {idx: bytes[idx] for idx in range(256)}
        tokens = list(text.encode('utf-8'))
        num_merges = vocab_size - 256
        for i in range(num_merges):
            counts = count_pairs_of_tokens(tokens)
            most_count_pair = max(counts, key=counts.get)
            new_idx = 256 + i
            tokens = merge(tokens, most_count_pair, new_idx)
            merges[most_count_pair] = new_idx
            vocab[new_idx] = vocab[most_count_pair[0]] + \
                vocab[most_count_pair[1]]  # sum of bytes

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = count_pairs_of_tokens(tokens)
            pair = min(stats, key=lambda x: self.merges.get(x, float('inf')))
            if self.merges.get(pair):
                tokens = merge(tokens, pair, self.merges[pair])
            else:
                return tokens

    def decode(self, sequence: List[int]) -> str:
        tokens = b"".join(self.vocab[idx] for idx in sequence)
        text = tokens.decode("utf-8", errors="replace")
        return text
