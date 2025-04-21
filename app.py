from flask import Flask , render_template , jsonify , request
from src.helper import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

embeddings = download_hugging_face_embeddings() 

#Initialize the Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "sbot"


pc.Index(index_name)

# Load Existing index 
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

#creating the prompt template
PROMPT = PromptTemplate(template=prompt_template, 
                        input_variables=["context", "question"])


chain_type_kwargs = {"prompt": PROMPT}

#initilizing the llm model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY , temperature=0.7)

#initializing the qa retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Uses "stuff" prompt pattern
    retriever=retriever,  # Make sure retriever is from PineconeVectorStore
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)