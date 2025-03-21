from transformers import AutoTokenizer
from collections import defaultdict


class BertTokenizer:
    """Word-piece tokenizer using BERT tokenizer."""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize(self, text: str) -> list[str]:
        """Tokenizes the input text.

        Args:
            text (str): Input text to tokenize.

        Returns:
            List[str]: List of tokens.
        """
        return self.tokenizer.tokenize(text)

    def encode(self, text: str) -> list[int]:
        """Encodes the input text.

        Args:
            text (str): Input text to encode.

        Returns:
            list[int]: List of token ids.
        """
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decodes the input tokens.

        Args:
            tokens (list[int]): List of token ids.

        Returns:
            str: Decoded text.
        """
        return self.tokenizer.decode(tokens)


# Example on how to use it
if __name__ == "__main__":
    # Word-piece tokenizer
    tokenizer = BertTokenizer()

    # Example
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    word_freqs = defaultdict(int)
    for text in corpus:
        tokens = tokenizer.tokenize(text)
        print(f"    Tokens: {tokens}")

        # Decoding process:
        recovered_text = tokenizer.decode(tokenizer.encode(text))
        print(f"    Recovered text: {recovered_text}")
        print(f"    Original text: {text}")
        print("-" * 50)
        for token in tokens:
            word_freqs[token] += 1
    print(f"Word frequencies: {word_freqs}")