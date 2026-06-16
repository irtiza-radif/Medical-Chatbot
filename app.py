import os
import socket
from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
from src.prompt import *

app = Flask(__name__)
load_dotenv()

# Secure all API Keys from .env ()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# Override proxy catches that disrupt urllib3 channels on Windows
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

# NETWORK PATCH: Windows DNS Resolution এবং সকেট ব্লকেড কাটানোর জন্য
try:
    socket.getaddrinfo('api.mistral.ai', 443)
    socket.getaddrinfo('api.pinecone.io', 443)
except socket.gaierror:
    print("Network environment restricted. Activating local resolution bypass...")

# 1. Initialize Embeddings & Pinecone Connection
print("Initializing local HuggingFace embedding engine...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'local_files_only': True}
)
print("Connecting to existing Pinecone Index...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# 2. Build the Tools for our Agent
print("Setting up Agent Tools (Pinecone Textbook Base Only)...")

medical_book_tool = create_retriever_tool(
    retriever,
    name="medical_knowledge_base",
    description="Searches and returns data directly from the official medical textbook. Use this as your primary source for verified medical definitions, symptoms, and textbook treatments."
)


tools = [medical_book_tool]

# 3. Initialize Mistral AI Model & Agent Prompt
print("Loading chat model framework (Mistral AI)...")
chatmodel = ChatMistralAI(model="mistral-large-latest", temperature=0.3)

agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful and expert medical assistant. You have access to a verified medical textbook tool. Always prioritize the medical textbook tool first to answer verified definitions, symptoms, and treatments. If the information is missing from the textbook, seamlessly use your own internal advanced knowledge base to formulate an accurate answer. Keep answers concise (max 3-4 sentences)."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 4. Construct the Intelligent Agent
agent = create_tool_calling_agent(chatmodel, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/get', methods=["GET", "POST"])
def chat():
    msg = request.form['msg']
    print(f"User Question: {msg}")

    try:
        # 1. Main pipeline execution (Textbook DB -> Mistral Internal Brain)
        response = agent_executor.invoke({"input": msg})
        raw_output = response.get("output", "")

    except Exception as e:
        print(f"Agent Engine Error caught: {e}")

        # 2. OPTIMIZED FALLBACK: মিস্ট্রাল এপিআই-এর রেট লিমিট (429) এরর হ্যান্ডলিং
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "rate limit" in str(e).lower():
            raw_output = "The system is currently handling too many requests under the Mistral API tier. Please wait a few moments and try sending your query again!"
        else:
            # Fallback for standard connectivity/database drops
            try:
                local_docs = retriever.invoke(msg)
                if local_docs:
                    raw_output = f"[Local Search Fallback] Here is the matching text found directly in your textbook: {local_docs[0].page_content}"
                else:
                    raw_output = "I ran into a connection issue. Please verify your internet settings and retry."
            except Exception as fallback_err:
                raw_output = "System connection busy. Please wait a moment before trying again."

    # 3. Clean up list structures if returned by the model
    if isinstance(raw_output, list) and len(raw_output) > 0:
        if isinstance(raw_output[0], dict) and 'text' in raw_output[0]:
            raw_output = raw_output[0]['text']

    print("Final Processed Response sent to UI: ", raw_output)
    return str(raw_output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
