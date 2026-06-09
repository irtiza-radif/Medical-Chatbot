from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
# <-- Fixed: Imported correct Gemini wrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# <-- Fixed: Imported directly here to stop NameError
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("Initializing local HuggingFace embedding engine...")
# By instantiating it directly here, it is impossible for Python to throw a NameError!
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

print("Connecting to existing Pinecone Index...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

print("Loading chat model framework...")
chatmodel = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answering_chain = create_stuff_documents_chain(chatmodel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)


@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/get', methods=["GET", "POST"])
def chat():
    msg = request.form['msg']
    input = msg
    print(f"User Question: {input}")
    response = rag_chain.invoke({"input": msg})
    print("Response: ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
