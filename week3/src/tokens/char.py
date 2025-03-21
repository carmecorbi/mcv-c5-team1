import numpy as np


class CharTokenizer:
    def __init__(self, vocab, special_chars, max_len=None):
        """
        Initialize the character tokenizer.
        
        Args:
            vocab (list): List of characters in the vocabulary
            special_chars (list): List of special characters [START, END, PAD]
            max_len (int, optional): Maximum sequence length
        """
        self.vocab = vocab
        self.chars = special_chars  # [START, END, PAD]
        self.max_len = max_len
        
        # Create mappings between characters and indices
        self.char2idx = {char: idx for idx, char in enumerate(vocab, start=0)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        
        print("Length of special_chars:", len(special_chars))
        print("Special chars:", special_chars)
        print("First few vocab items:", vocab[:5])
    
    def encode(self, text):
        """
        Encode text to token indices with special characters.
        
        Args:
            text: Input text to encode (str or list/array of str)
            
        Returns:
            List or numpy array of character indices
        """
        # Handle batch input
        if isinstance(text, (list, np.ndarray)):
            return np.array([self._encode_single(t) for t in text])
        # Handle single input
        else:
            return self._encode_single(text)
    
    def _encode_single(self, text):
        """Helper method to encode a single text input"""
        # Convert text to character list
        char_list = list(text)
        
        # Add special characters
        final_list = [self.chars[0]]  # Start token
        final_list.extend(char_list)
        final_list.append(self.chars[1])  # End token
        
        # Handle padding if max_len is specified
        if self.max_len is not None:
            gap = self.max_len - len(final_list)
            if gap > 0:
                final_list.extend([self.chars[2]] * gap)  # Add padding tokens
            elif gap < 0:
                # Truncate if exceeding max length
                final_list = final_list[:self.max_len]
        
        # Convert characters to indices
        return [self.char2idx[char] for char in final_list]
    
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
        # Convert indices to characters
        chars = [self.idx2char[idx] for idx in indices]
        
        # Find end token position
        try:
            end_idx = chars.index(self.chars[1])
            chars = chars[:end_idx]  # Remove everything after end token
        except ValueError:
            pass
        
        # Remove start token if present
        if chars and chars[0] == self.chars[0]:
            chars = chars[1:]
        
        # Join characters and return text
        return ''.join(chars)
    
    def __len__(self):
        return len(self.vocab)