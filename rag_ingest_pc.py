from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

import os

load_dotenv(override=True)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Get path of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct path to sample.txt
file_path = os.path.join(BASE_DIR, "packages", "dtsense-rag", "dtsense_rag", "data", "harry_potter_knowledge.txt")

# Langkah 1: Mengambil konten dari halaman web menggunakan WebBaseLoader
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
documents = text_splitter.split_documents(documents)

# Langkah 2: Menggunakan Hugging Face Embeddings untuk membuat vektor dari dokumen
# Menggunakan model Hugging Face untuk embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/gtr-t5-base")

index_name = "dts-project-data"
# Cek apakah index sudah ada, jika tidak buat index baru
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,  # Sesuaikan dengan dimensi model embedding yang digunakan
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",  # Ganti dengan region yang sesuai
        )
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)
result = vector_store.add_documents(documents)
print(f"Added {len(result)} documents to Pinecone index '{index_name}'.")