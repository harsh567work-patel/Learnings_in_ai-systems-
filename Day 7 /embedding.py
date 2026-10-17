import os 
from pathlib import Path
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
text =  "machine learning is fun"

def cosine_similarity(a,b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



t1 = " There are 21 paid leaves"
t2 = " there are 24 vacations "

v1 = model.encode(t1)
v2 = model.encode(t2)
print(cosine_similarity(v1, v2)) 

# HOW THIS HELPS:
# This code snippet demonstrates how to compute the cosine similarity between two text strings using sentence embeddings.
