import os
from dotenv import load_dotenv

load_dotenv()


PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME=os.getenv("PINECONE_INDEX_NAME","agentic-ai-index")

 

   
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM=384
LLM_MODEL="llama3.2:3b"

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=3

RELEVANCE_THRESHOLD = 0.25
TOP_K = 3
GROUNDING_THRESHOLD = 0.35
