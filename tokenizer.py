# -*- coding: utf-8 -*-
"""Tokenizer.ipynb

#**Generating Your First Text**
"""

!pip install -q transformers==4.48.3 accelerate sentencepiece
import transformers
print(transformers.__version__)

from transformers import AutoModelForCausalLM, AutoTokenizer
#Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    revision="main"
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", revision="main")

from transformers import pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False,
    temperature=0.7
)

# The prompt
messages = [
    {"role": "user",
     "content": "Create a funny joke about India."
     }
]

# Generate output
output = generator(messages)
print(output[0]["generated_text"])

"""# **Tokenizer**"""

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. <|assistance|>"

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Generate the text
generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=20
)

# Print the output
print(tokenizer.decode(generation_output[0]))

# Prompt's/Input token IDs
print(input_ids)

# Decode the prompt/Input from their token ids.
for id in input_ids[0]:
  print(tokenizer.decode(id))

# input tokens as well as the output tokens ids.
print(generation_output)

print(tokenizer.decode(16423))
print(tokenizer.decode(292))
print(tokenizer.decode([16423,292]))

"""#**Comparing Trained LLM Tokenizers**

### 1. BERT Tokenizer (WordPiece)
"""

# !pip install transformers

from transformers import AutoTokenizer

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Test sentence
text = text = """
Hello     ChatGPT!

I love NLP 😊.
Email: training123@gmail.com

Price = $45.67
Version=v2.5.1
snake_case_variable
camelCaseVariable
C++, Python, Java!
भारत 日本 العربية
"""

print("=" * 60)
print("Original Text:")
print(text)

# Tokenize
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\n" + "=" * 60)
print("Tokens:")
for i, token in enumerate(tokens):
    print(f"{i:2d}: {repr(token)}")

print("\n" + "=" * 60)
print("Token IDs:")
print(token_ids)

print("\n" + "=" * 60)
print("Token -> ID Mapping:")
for token, tid in zip(tokens, token_ids):
    print(f"{repr(token):20} -> {tid}")

# Encode with special tokens
encoded = tokenizer.encode(text, add_special_tokens=True)

print("\n" + "=" * 60)
print("Encoded IDs (with special tokens):")
print(encoded)

print("\nDecoded back:")
print(tokenizer.decode(encoded))

print("\n" + "=" * 60)
print("Special Tokens:")
print("CLS Token :", tokenizer.cls_token)
print("SEP Token :", tokenizer.sep_token)
print("PAD Token :", tokenizer.pad_token)
print("UNK Token :", tokenizer.unk_token)
print("MASK Token:", tokenizer.mask_token)

"""###2. GPT Tokenizer (using tiktoken)"""

# !pip install tiktoken

import tiktoken

# Load GPT tokenizer (used by many OpenAI models)
encoding = tiktoken.get_encoding("cl100k_base")

text = """
Hello, world!     This is a tokenizer test.

Email: john.doe@example.com
Price: $19.99
Emoji: 😊
Python_version = 3.11
C++ > Java?
"""

print("=" * 60)
print("Original Text:")
print(text)

# Encode
token_ids = encoding.encode(text)

print("\n" + "=" * 60)
print("Token IDs:")
print(token_ids)

print("\n" + "=" * 60)
print("Token Breakdown:")

for i, token_id in enumerate(token_ids):
    token_bytes = encoding.decode_single_token_bytes(token_id)

    try:
        token_text = token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        token_text = str(token_bytes)

    print(f"{i:2d}: ID={token_id:<8} Token={repr(token_text)}")

print("\n" + "=" * 60)
print("Decoded Text:")
print(encoding.decode(token_ids))

print("\n" + "=" * 60)
print("Total Tokens:", len(token_ids))

"""###3. StarCoder Tokenizer
Unlike BERT (optimized for natural language) and GPT's general-purpose tokenizer, StarCoder's tokenizer is trained heavily on source code, so it often treats code patterns, operators, indentation, and identifiers more naturally.
"""

!pip install transformers sentencepiece

from transformers import AutoTokenizer
import huggingface_hub
huggingface_hub.login(token="hf_access_token") # Replace the hf_access_token

# Load StarCoder tokenizer
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")

# Test text containing both natural language and code
text = text = """
for i in range(10):
    print(i)

my_variable = "Hello, World!"
total_price = 45.67

if x >= 100 and y != 0:
    result += x // y

# Comment
user_email = "training123@gmail.com"

C++
Python3.11
snake_case
camelCase
😊
"""

print("=" * 60)
print("Original Text:")
print(text)

# Tokenize
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\n" + "=" * 60)
print("Tokens:")
for i, token in enumerate(tokens):
    print(f"{i:2d}: {repr(token)}")

print("\n" + "=" * 60)
print("Token IDs:")
print(token_ids)

print("\n" + "=" * 60)
print("Token -> ID Mapping:")
for token, tid in zip(tokens, token_ids):
    print(f"{repr(token):30} -> {tid}")

# Encode
encoded = tokenizer.encode(text)

print("\n" + "=" * 60)
print("Encoded IDs:")
print(encoded)

print("\nDecoded back:")
print(tokenizer.decode(encoded))

print("\n" + "=" * 60)
print("Total Tokens:", len(tokens))

"""###4. Llama Tokenizer (SentencePiece)"""

!pip install transformers sentencepiece

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

text = """
Hello, world!

Email: john.doe@example.com
Price: $19.99
Emoji: 😊
Python_version = 3.11
C++ > Java?
"""

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("=" * 60)
print("Original Text:")
print(text)

print("\nTokens:")
for i, token in enumerate(tokens):
    print(f"{i:2d}: {repr(token)}")

print("\nToken -> ID Mapping:")
for token, tid in zip(tokens, token_ids):
    print(f"{repr(token):25} -> {tid}")

print("\nDecoded:")
print(tokenizer.decode(token_ids))

"""###5. Qwen Tokenizer"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    trust_remote_code=True
)

text = """
Hello     ChatGPT!

Email: training123@gmail.com
Price = $45.67

snake_case_variable
camelCaseVariable

C++ >= Java?
😊

भारत 日本 العربية
"""

print("=" * 60)
print("Original Text:")
print(text)

# Tokenize
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\n" + "=" * 60)
print("Tokens:")
for i, token in enumerate(tokens):
    print(f"{i:2d}: {repr(token)}")

print("\n" + "=" * 60)
print("Token -> ID Mapping:")
for token, tid in zip(tokens, token_ids):
    print(f"{repr(token):30} -> {tid}")

print("\n" + "=" * 60)
print("Decoded:")
print(tokenizer.decode(token_ids))

print("\nTotal Tokens:", len(token_ids))

