from typing import List, Tuple
from abc import abstractmethod, ABC


def count_pairs_of_tokens(ids: List[int], counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = dict() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(tokens: List[int], pair: Tuple[int], new_idx: int):
    i = 0
    to_ret = []
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            to_ret.append(new_idx)
            i += 2
        else:
            to_ret.append(tokens[i])
            i += 1
    return to_ret


class Tokenizer(ABC):
    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    @abstractmethod
    def train(self, text: str, vocab_size: int):
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass

    def _build_vocab(self):
        vocab = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for pair, index in self.merges.items():
            first, second = pair[0], pair[1]
            vocab[index] = vocab[first] + pair[second]
        for token, idx in self.special_tokens.items():
            vocab[idx] = token.encode("utf-8")
        return vocab
