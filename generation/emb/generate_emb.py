import json
import openai
import numpy as np

openai.api_key = "" # YOUR OPENAI API_KEY

def get_gpt_emb(prompt):

    embedding = openai.Embedding.create(
        input=prompt,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    return np.array(embedding)

# Read generated profiles
profiles = []
with open('./generation/emb/profiles.json', 'r') as f:
    for line in f.readlines():
        profiles.append(json.loads(line))

class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

print(Colors.GREEN + "Encoding Semantic Representation" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The Profile is:\n" + Colors.END)
print(profiles[0]['summarization'])
print("---------------------------------------------------\n")
emb = get_gpt_emb(profiles[0]['summarization'])
print(Colors.GREEN + "Encoded Semantic Representation Shape:" + Colors.END)
print(emb.shape)
