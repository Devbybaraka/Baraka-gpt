class CharTokenizer:
    def __init__(self,text):

        # Get all uniques characters and sort them
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        #create 2 dictionaries for translation 
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        print(f"Tokenizer created. found {self.vocab_size} unique characters.")
        print(f"Characters: {self.chars}")

    def encode(self, s):
        #take as string, return lis of numbers 
        # Foe each character in string, look up its number

        return[self.stoi[c] for c in s ]

    def decode(self, nums):

        return''.join([self.itos[i] for  i in nums ])

    def save(self, filepath = "tokenizer_vocab.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        print(f"Vocab loaded. size: {self.vocab_size}")
        return self

if __name__ == "__main__" :
    with open("war_and_peace.txt", "r", encoding="utf-8") as f :
        book_text = f.read()

    print(f"Book loaded. it has {len(book_text)} characters.")
     
    tokenizer = CharTokenizer(book_text)
    test_sentence = "Hello"
    numbers = tokenizer.encode(test_sentence)
    print(f"\nEncode test:")
    print(f"'{test_sentence}' -> {numbers}")

    back_to_text = tokenizer.decode(numbers)
    print(f"\nEncode test:")
    print(f"{numbers} -> '{back_to_text}'")

    original = "prince"
    encode = tokenizer.encode(original)
    decode = tokenizer.decode(encode)
    print(f"\nRound trip:")
    print(f"Original: {original}")
    print(f"Encoded: {encode}")
    print(f"decoded:{decode}")
    print(f"Succes:{original == decode}")
    
    tokenizer.save()



    
            
