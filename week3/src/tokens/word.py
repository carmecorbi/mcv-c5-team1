from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from src.dataset.prepare_data import load_data
from collections import defaultdict
from typing import Union, List

import numpy as np
import re


class WordTokenizer:
    def __init__(self, vocab, special_chars, max_len: int = 201):
        """
        Initialize the word tokenizer.
        
        Args:
            vocab (list): List of words in the vocabulary
            special_words (list): List of special words [START, END, PAD]
            max_len (int, optional): Maximum sequence length
        """
        self.vocab = vocab
        self.words = special_chars  # [START, END, PAD]
        self.max_len = max_len
        
        # Create mappings between characters and indices
        self.word2idx = {word: idx for idx, word in enumerate(vocab, start=0)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print("Length of special_words:", len(special_chars))
        print("Special words:", special_chars)
        print("First few vocab items:", vocab[:20])

    def encode(self, text):
        """
        Encode text to token indices with special characters.
        
        Args:
            text: Input text to encode (str or list/array of str)
            
        Returns:
            List or numpy array of word indices
        """
        # Handle batch input
        if isinstance(text, (list, np.ndarray)):
            return np.array([self._encode_single(t) for t in text])
        # Handle single input
        else:
            return self._encode_single(text)

    def _encode_single(self, text):
        """Helper method to encode a single text input"""
        # Convert text to word list
        segm_tokens = re.split(r'(\s+)', text)
        word_list = []
        for segm in segm_tokens:
            if segm.isspace():
                word_list.append(segm)
            else:
                word_list.extend(word_tokenize(segm))
        
        # Add special characters
        final_list = [self.words[0]]  # Start token
        final_list.extend(word_list)
        final_list.append(self.words[1])  # End token
        
        # Handle padding if max_len is specified
        if self.max_len is not None:
            gap = self.max_len - len(final_list)
            if gap > 0:
                final_list.extend([self.words[2]] * gap)  # Add padding tokens
            elif gap < 0:
                # Truncate if exceeding max length
                final_list = final_list[:self.max_len]
        
        # Convert characters to indices
        return [self.word2idx[word] for word in final_list]

    def decode(self, indices):
        """
        Decode indices back to text.
        
        Args:
            indices: List/array of token indices or batch of token indices
            
        Returns:
            str or list of str: Decoded text with special tokens removed
        """
        # Handle batch input
        if isinstance(indices, (list, np.ndarray)) and len(indices) > 0:
            # Check if this is a batch (2D array) or single sequence
            if isinstance(indices[0], (list, np.ndarray)):
                return [self._decode_single(seq) for seq in indices]
            else:
                return self._decode_single(indices)
        else:
            return ""
    
    def _decode_single(self, indices):
        """Helper method to decode a single sequence"""
        # Convert indices to words
        words = [self.idx2word[idx] for idx in indices]
        
        # Find end token position
        try:
            end_idx = words.index(self.words[1])
            words = words[:end_idx]  # Remove everything after end token
        except ValueError:
            pass
        
        # Remove start token if present
        if words and words[0] == self.words[0]:
            words = words[1:]
        
        # Join characters and return text
        return ''.join(words)
    
    def __len__(self):
        return len(self.vocab)


# Example on how to use it
if __name__ == "__main__":

    csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/raw_data.csv'

    # Load data
    df = load_data(csv_path)
    special_chars = ['<SOS>', '<EOS>', '<PAD>']

    # Extract unique words from the 'Title' column
    vocab = set()

    # Loop over each caption in the 'Title' column
    for caption in df['Title']:
        # Everything is included (punctuation, uppercase, lowercase)
        segm_tokens = re.split(r'(\s+)', caption)
        word_list = []
        for segm in segm_tokens:
            if segm.isspace():
                word_list.append(segm)
            else:
                word_list.extend(word_tokenize(segm))
        vocab.update(word_list)

    vocab_list = special_chars + list(vocab)

    # Get the tokenizer
    word_tokenizer = WordTokenizer(vocab_list, special_chars=special_chars, max_len=50)

    # Example
    corpus = [
        "Rice with Soy-Glazed Bonito Flakes and Sesame Seeds",
        "Mrs. Marshall's Cheesecake",
        "Mole Cake with Cherry-Almond Ice Cream, Tamarind Anglaise, and Orange Caramel",
        "Doenjang Jjigae (된장찌개 / Fermented-Soybean Stew)",
    ]

    word_freqs = defaultdict(int)

    for text in corpus:
        tokens = word_tokenizer.encode(text)
        print(f"    Tokens: {tokens}")

        # Decoding process:
        recovered_text = word_tokenizer.decode(tokens)
        print(f"    Recovered text: {recovered_text}")
        print(f"    Original text: {text}")
        print("-" * 50)
        for token in tokens:
            word_freqs[token] += 1
    print(f"Word frequencies: {word_freqs}")
    print(f"Number of tokens: {len(word_tokenizer)}")
    
    # Can also be used with a batch of input
    token_batch = word_tokenizer.encode(corpus)
    print(f"Token batch: {token_batch}")
    print(f"Decoded batch: {word_tokenizer.decode(token_batch)}")
    print(f"Original batch: {corpus}")
    