from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings  # Imported directly here!

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("1. Loading PDF files from data/ directory...")
extracted_data = load_pdf_file(data='data/')

print("2. Filtering metadata...")
filter_data = filter_to_minimal_docs(extracted_data)

print("3. Splitting text into smaller chunks...")
texts_chunk = text_split(filter_data)

print("4. Initializing Hugging Face Embedding model directly...")
# By defining it here, it is impossible for Python to say it is not defined!
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

print("5. Initializing Pinecone Client...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# Check and safely create the index if it doesn't exist
if not pc.has_index(index_name):
    print(f"Index '{index_name}' not found. Creating a new one...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

print("6. Vectorizing chunks and uploading directly to Pinecone (This may take a minute)...")
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    index_name=index_name,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

print("✅ Success! Your Pinecone vector store is fully populated and ready.")
