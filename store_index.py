from src.helper import load_pdf , text_split, download_hugging_face_embeddings
import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

extracted_data = load_pdf("data/")   #Get the data from the data folder
text_chunks = text_split(extracted_data)   #Split the data into smaller chunks
embeddings = download_hugging_face_embeddings()   #Download the embeddings model

#Initialize pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "sbot"
pc.Index(index_name)

#embed each chunk and upsert the embeddings into pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding = embeddings,
    index_name = index_name
)
