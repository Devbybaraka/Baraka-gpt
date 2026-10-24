class CharTokenizer:
    def __init__(self, text_path):
        # 1. Read the text file
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 2. Get all unique characters in the text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # 3. Create mapping dictionaries (String to Integer & Integer to String)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        print(f"Loaded dataset from {text_path}")
        print(f"Vocabulary Size: {self.vocab_size} unique characters")

    def encode(self, s):
        """ Takes a string, outputs a list of integers """
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l):
        """ Takes a list of integers, outputs a string """
        return ''.join([self.itos[i] for i in l])


# --- Quick Test Block ---
if __name__ == "__main__":
    # Create a small dummy text file for testing
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write("hello world! this is my first tokenizer test.")

    # Initialize tokenizer
    tokenizer = CharTokenizer("input.txt")

    # Test encoding
    sample_text = "hello world"
    encoded_ids = tokenizer.encode(sample_text)
    print(f"\nOriginal text: '{sample_text}'")
    print(f"Encoded IDs:   {encoded_ids}")

    # Test decoding
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"Decoded text:  '{decoded_text}'")