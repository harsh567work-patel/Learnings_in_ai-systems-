import os 
from dotenv import Qdrantclient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance , VectorParams , PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = QdrantClient(
    url = QDRANT_URL,
    api_key = QDRANT_API_KEY
)

print("Qdrant client initialized successfully.")

COLLECTION_NAME = "knowledge_base"
EMBEDDING_SIZE = 384

if client.collection_exists(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' already exists.")
    client.delete_collection(COLLECTION_NAME)

    client.create_collection(
     collection_name = COLLECTION_NAME,
     vectors_config = VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE)
 ) 

print(f"Collection '{COLLECTION_NAME}' created successfully.")
print(f"Vector size: {EMBEDDING_SIZE}, Distance metric: COSINE")
print("DISTANCE METRIC: COSINE")

with open( "knowledge.txt" , "r" , encoding="utf-8") as f: # this makes text lines in array structure 
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loading ready.")

embeddings = model.encode(documents)





