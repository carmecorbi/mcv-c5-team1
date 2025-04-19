from transformers import AutoTokenizer
from collections import defaultdict
from typing import Union, List

import numpy as np


class Tokenizer:
    def __init__(self, tokenizer_path: str = "bert-base-cased", max_length: int = 60):
        """Word-piece tokenizer using BERT tokenizer.

        Args:
            max_length (int, optional): Max length of the tokenization. Defaults to 201.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length

    def tokenize(self, text: str) -> list[str]:
        """Tokenizes the input text.

        Args:
            text (str): Input text to tokenize.

        Returns:
            List[str]: List of tokens.
        """
        return self.tokenizer.tokenize(text)

    def encode(self, text: str | List[str]) -> list[int] | List[list[int]]:
        """Encodes the input single text or a batch of inputs.

        Args:
            text (str or List[str]): Input text to encode or batch of texts to encode.

        Returns:
            list[int] or List[list[int]]: List of token ids or list of token id sequences for a batch.
        """
        return self.tokenizer(text, add_special_tokens=True, padding='max_length', max_length=self.max_length)['input_ids']

    def decode(self, tokens: list[int] | List[list[int]], skip_special_tokens: bool=True) -> Union[str, List[str]]:
        """Decodes either a single sequence or batch of token sequences.

        Args:
            tokens: Either a list of token ids [int] or a batch of token sequences [[int]].
            skip_special_tokens (bool): Whether to remove special tokens like [CLS], [SEP], etc.

        Returns:
            str or List[str]: Decoded text(s). Returns a string for a single sequence
                            or a list of strings for a batch.
        """
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
            
        # Check if tokens is a batch (list of lists) or a single sequence
        if len(tokens) > 0 and isinstance(tokens[0], list):
            # It's a batch - use batch_decode
            return self.tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens)
        else:
            # It's a single sequence - use decode
            return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def __len__(self):
        return len(self.tokenizer)
    
    def get_special_tokens(self):
        return self.tokenizer.special_tokens_map
    
    def get_special_token_ids(self):
        return self.tokenizer.all_special_ids


# Example on how to use it
if __name__ == "__main__":
    # Word-piece tokenizer
    tokenizer = Tokenizer("nlpconnect/vit-gpt2-image-captioning")

    # Example
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    word_freqs = defaultdict(int)
    for text in corpus:
        tokens = tokenizer.encode(text)
        print(f"    Tokens: {tokens}")

        # Decoding process:
        recovered_text = tokenizer.decode(tokenizer.encode(text))
        print(f"    Recovered text: {recovered_text}")
        print(f"    Original text: {text}")
        print("-" * 50)
        for token in tokens:
            word_freqs[token] += 1
    print(f"Word frequencies: {word_freqs}")
    print(f"Number of tokens: {len(tokenizer)}")
    
    # Can also be used with a batch of input
    token_batch = tokenizer.encode(corpus)
    print(f"Token batch: {token_batch}")
    print(f"Decoded batch: {tokenizer.decode(token_batch)}")
    print(f"Original batch: {corpus}")
    
    print(f"Special tokens: {tokenizer.get_special_tokens()}")
    print(f"Special token ids: {tokenizer.get_special_token_ids()}")