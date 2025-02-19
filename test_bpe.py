"""
https://sebastianraschka.com/blog/2025/bpe-from-scratch.html

1. Identify frequent pairs

In each iteration, scan the text to find the most commonly occurring pair of bytes (or characters)
2. Replace and record

Replace that pair with a new placeholder ID (one not already in use, e.g., if we start with 0…255, the first placeholder would be 256)
Record this mapping in a lookup table
The size of the lookup table is a hyperparameter, also called “vocabulary size” (for GPT-2, that’s 50,257)
3. Repeat until no gains

Keep repeating steps 1 and 2, continually merging the most frequent pairs
Stop when no further compression is possible (e.g., no pair occurs more than once)
Decompression (decoding)

To restore the original text, reverse the process by substituting each ID with its corresponding pair, using the lookup table



t h e ^ c a t ^ i n ^ t h e ^ h a t h
0 1 2 3 4 5 0 3 6 7 3 0 1 2 3 1 5 0 1

8 e ^ c a t ^ i n ^ 8 e ^ h a 8
8 2 3 4 5 0 3 6 7 3 8 2 3 1 5 8

token -> id
0 1 -> 8


0 1 -> 3
1 2 -> 2
e ^ -> 2

okay so I think I'd be able to do this...I just need to maintain a list of indexes...
in a linkedlist

and then I'd be able to do the replacement quickly
but maybe this is over optimizing? I should just go over the string multiple times??

wait how is this going to work...lol...this is very confusing...

(1) e c a (1) e

(th)(he)(ec)(ca)(at)(th)(he)


I need to find all the nodes...pop them out...AND their neighbors...and then insert 
the nodes back in after mutating the first part of the BPE...
AND I need to be updating the counter...
but then I run into the CLASSIC issue of how to update a heap inside of itself...
(1e)(ec)(ca)(at)(1e)

https://www.youtube.com/watch?v=zduSFxRajkE

TODO: how to encode [UNK]??

"""
import pytest
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import heapq

class MaxHeap:
    def __init__(self):
        self.h = []

    def push(self, k, v):
        heapq.heappush(self.h, (-v, k))

    def pop(self):
        negv, k = heapq.heappop(self.h)
        return -negv, k
    
    def peek(self):
        return self.h[0]

        
A_ORD = ord('a')

class Node:
    def __init__(self, key, p=None, n=None):
        self.key = key
        # self.val = val
        self.p: Node = p
        self.n: Node = n

class LL:
    def __init__(self):
        self.head = Node(-1,-1)
        self.tail = Node(-1,-1)
        self.head.next = self.tail
        self.tail.prev = self.head
        
    def remove(self, node: Node):
        node.n.p = node.p
        node.p.n = node.n
        node.p, node.n = None, None
    
    def insert_after(self, curr: Node, new_node: Node):
        new_node.n = curr.n
        new_node.p = curr
        curr.n.p = new_node
        curr.n = new_node

    def append(self, node: Node):
        node.p = self.tail.p
        node.n = self.tail
        self.tail.p.n = node
        self.tail.p = node
        
        
class BPETokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}

    def train(self, texts: List[str]) -> None:
        """Train the BPE tokenizer on a list of texts."""
        ll = LL()
        counter = Counter()
        key_to_nodes = defaultdict(lambda: set())
        for text in texts:
            for i in range(len(text)-1):
                bp = ord(text[i]), ord(text[i])
                node = Node(key=bp)
                ll.append(node)
                key_to_nodes[bp].add(node)
                counter[bp] += 1
        maxheap = MaxHeap()
        for pair, count in counter.items():
            maxheap.push(pair, count)
        while maxheap and maxheap.peek() > 1 and len(self.vocab) < self.vocab_size:
            count, (t1, t2) = maxheap.pop()
            pass
            
        
        raise NotImplementedError()

    def encode(self, text: str) -> List[int]:
        """Encode text into token ids using learned BPE merges."""
        raise NotImplementedError()

    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids back into text."""
        raise NotImplementedError()

@pytest.fixture
def tokenizer():
    return BPETokenizer(vocab_size=100)

@pytest.fixture
def trained_tokenizer(tokenizer):
    training_texts = [
        "hello world",
        "hello there",
        "world peace"
    ]
    tokenizer.train(training_texts)
    return tokenizer

def test_vocab_size_limit(trained_tokenizer):
    """Test that the vocabulary size doesn't exceed the specified limit."""
    assert len(trained_tokenizer.vocab) <= trained_tokenizer.vocab_size

def test_base_characters_in_vocab(trained_tokenizer):
    """Test that all base characters from training data are in the vocabulary."""
    training_chars = set("hello world there peace")
    for char in training_chars:
        assert char in trained_tokenizer.vocab

@pytest.mark.parametrize("test_text", [
    "hello world",
    "testing one two three",
    "byte pair encoding",
    "a b c d e f g"
])
def test_roundtrip_consistency(tokenizer, test_text):
    """Test that encoding and then decoding returns the original text."""
    tokenizer.train([test_text])
    token_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(token_ids)
    assert decoded_text == test_text

def test_subword_tokenization(tokenizer):
    """Test that the tokenizer properly handles subword units."""
    training_texts = [
        "playing played player plays",
        "running runner ran runs"
    ]
    test_text = "playing"
    
    tokenizer.train(training_texts)
    tokens = tokenizer.encode(test_text)
    
    # Verify that common subwords are tokenized together
    assert len(tokens) < len(test_text), "No subword merging occurred for common patterns"

@pytest.mark.parametrize("training_text,test_text", [
    (["hello world"], "goodbye world"),
    (["the cat"], "the dog"),
    (["running fast"], "running slow")
])
def test_unknown_tokens(tokenizer, training_text, test_text):
    """Test handling of characters/subwords not seen during training."""
    tokenizer.train(training_text)
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    assert decoded == test_text

def test_empty_input(trained_tokenizer):
    """Test handling of empty input for both encode and decode."""
    assert trained_tokenizer.encode("") == []
    assert trained_tokenizer.decode([]) == ""

def test_merge_rules_common_pairs(tokenizer):
    """Test that merge rules are being learned for common pairs."""
    training_text = ["aaaa"]  # Should learn to merge 'aa' pairs
    tokenizer.train(training_text)
    assert ('a', 'a') in tokenizer.merges, "Failed to learn obvious merge rule for repeated character"

@pytest.mark.parametrize("text,expected_token_count", [
    ("hello", 5),  # Before training, should be character-level
    ("a" * 10, 10),  # Before training, should be character-level
])
def test_initial_tokenization(tokenizer, text, expected_token_count):
    """Test initial tokenization before any training."""
    tokens = tokenizer.encode(text)
    assert len(tokens) == expected_token_count

def test_deterministic_encoding(trained_tokenizer):
    """Test that encoding the same text multiple times produces the same tokens."""
    text = "hello world"
    first_encoding = trained_tokenizer.encode(text)
    second_encoding = trained_tokenizer.encode(text)
    assert first_encoding == second_encoding

@pytest.mark.parametrize("special_chars", [
    "\n\t",  # Whitespace characters
    "!@#$%",  # Special characters
    "😀🌍🌞",  # Emoji
])
def test_special_characters(tokenizer, special_chars):
    """Test handling of special characters and emoji."""
    tokenizer.train([special_chars])
    tokens = tokenizer.encode(special_chars)
    decoded = tokenizer.decode(tokens)
    assert decoded == special_chars

def test_merge_priority(tokenizer):
    """Test that more frequent pairs are merged before less frequent ones."""
    training_text = ["aaaa bbbb aaaa"]  # 'aa' appears more times than 'bb'
    tokenizer.train([training_text])
    
    aa_merge_id = tokenizer.merges.get(('a', 'a'))
    bb_merge_id = tokenizer.merges.get(('b', 'b'))
    
    assert aa_merge_id is not None and bb_merge_id is not None
    assert aa_merge_id < bb_merge_id, "More frequent pair should be merged first"

def test_vocabulary_consistency(tokenizer):
    """Test that vocabulary remains consistent after multiple training iterations."""
    texts = ["hello world", "hello there"]
    
    tokenizer.train([texts[0]])
    vocab_after_first = tokenizer.vocab.copy()
    
    tokenizer.train([texts[1]])
    
    # Check that original vocabulary entries remain unchanged
    for token, id_ in vocab_after_first.items():
        assert tokenizer.vocab[token] == id_