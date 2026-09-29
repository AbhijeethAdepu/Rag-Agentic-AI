from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from pinecone import Pinecone,ServerlessSpec
import time
from src import config


def pdf():
    if os.path.exists(config.PDF_PATH):
        print(f"PDF already at {config.PDF_PATH}")
        return 
    os.makedirs(os.path.dirname(config.PDF_PATH),exist_ok=True)

def agentic_index(pc:Pinecone):
    name=config.PINECONE_INDEX_NAME
    if name not in [i.name for i in pc.list_indexes()]:
        print(f"Creating Index '{name}'.")
        pc.create_index(
            name=name,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws",region="us-east-1"),

        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
    else:
        print(f"Index '{name}' already exists")

def run_ingestion():
    
    pdf()
    pc=Pinecone(api_key=config.PINECONE_API_KEY)
    agentic_index(pc)

    # 1. Load document

    loader = PyPDFLoader(config.PDF_PATH)
    docs = loader.load()
    print(f"loaded {len(docs)} pages")

    # 2. Chunk document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    index=pc.Index(config.PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True)
        print("cleared old vectors")
    except Exception:
        pass

    # 3. Create embeddings & save to Pinecone
    embeddings =HuggingFaceEmbeddings(model=config.EMBED_MODEL,encode_kwargs={"normalize_embeddings":True},
                                              )
    vector_store = PineconeVectorStore(
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,

         
    )
    BATCH_SIZE=100
     

    for start in range(0,len(chunks), BATCH_SIZE):
        batch=chunks[start:start + BATCH_SIZE]
        ids = [f"chunk-{i}" for i in range(start, start+len(batch))]
        vector_store.add_documents(batch, ids=ids)
        print(f"Upserted chunks {start}-{start + len(batch)-1}")
                
    print("Ingestion complete")
    return vector_store     

if __name__=="__main__":
    run_ingestion()