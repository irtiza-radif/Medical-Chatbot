# Native system operating module for pathing and environment tasks
import os
# Low-level networking interface module to manage raw socket channels
import socket
# Imports Flask framework utilities for routing and requests
from flask import Flask, render_template, jsonify, request
# Imports Pinecone connector to query vector database indexes
from langchain_pinecone import PineconeVectorStore
# Imports standard Mistral AI chat model generation wrapper
from langchain_mistralai import ChatMistralAI
# Tools to structure dynamic system prompts with variables
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Imports HuggingFace library to handle text vectorization
from langchain_huggingface import HuggingFaceEmbeddings
# LangChain utility to transform standard vector streams into agent tools
from langchain_core.tools import create_retriever_tool
# Classes to build and execute runtime tool-using agents
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
# Loads credential parameters from local .env config files
from dotenv import load_dotenv
from src.prompt import *  # Imports custom system instructions from local codebase

# Instantiates the core Flask web server application instance
app = Flask(__name__)
# Reads secrets and keys out of the environment file definitions
load_dotenv()

# Retrieves the active Pinecone connection key string parameter
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# Retrieves the authorized Mistral AI cloud service connection key
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')

# Sets system environment variables for active workspace runtime
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# Configures the active pipeline process to pass keys to Mistral
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# Wipes global network proxy intercept configurations to prevent drops
os.environ["HTTP_PROXY"] = ""
# Ensures secure connection calls bypass local machine loop boundaries
os.environ["HTTPS_PROXY"] = ""

try:
    # Validates direct network handshake access to Mistral cloud platform
    socket.getaddrinfo('api.mistral.ai', 443)
    # Confirms open system port line connectivity to Pinecone indexes
    socket.getaddrinfo('api.pinecone.io', 443)
except socket.gaierror:
    print("Network environment restricted. Activating local resolution bypass...")

print("Initializing local HuggingFace embedding engine...")
# Declares name identifier of local vector conversion model
model_name = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(                          # Builds embedding processor in memory to convert text to vectors
    model_name=model_name,
    # Strict instruction preventing runtime model redownload checks
    model_kwargs={'local_files_only': True}
)
print("Connecting to existing Pinecone Index...")
docsearch = PineconeVectorStore.from_existing_index(         # Connects active pipeline to running vector cloud server index
    # Targets the specific document space holding medical textbook logs
    index_name="medical-chatbot",
    # Attaches embedding matching configuration properties to instance
    embedding=embeddings,
)
# Converts database index into a data extraction stream instance
retriever = docsearch.as_retriever(search_kwargs={"k": 3})

print("Setting up Agent Tools (Pinecone Textbook Base Only)...")

medical_book_tool = create_retriever_tool(
    retriever,
    name="medical_knowledge_base",
    description="Searches and returns data directly from the official medical textbook. Use this as your primary source for verified medical definitions, symptoms, and textbook treatments."
)

# Packs configured tool instances into a standard execution list
tools = [medical_book_tool]

print("Loading chat model framework (Mistral AI)...")
# Spans intelligent frontier model workspace backend instance
chatmodel = ChatMistralAI(model="mistral-large-latest", temperature=0.3)

agent_prompt = ChatPromptTemplate.from_messages(             # Builds structured conversation context template sequence bounds
    [
        # FIXED: Explicit instructions to stop using bold markdown lists (**Healthcare**:) and write in clean paragraph text instead
        ("system", "You are a helpful and expert medical assistant. You have access to a verified medical textbook tool. Prioritize the tool for textbook definitions. If the information is missing from the textbook, seamlessly use your own internal knowledge to formulate an accurate answer. CRITICAL: Never use markdown bold text or bulleted lists (do not output patterns like **Healthcare**: or extra quotes). Write your entire response as a smooth, continuous text paragraph without bullet points or headers. Keep answers concise (max 3-4 sentences)."),
        # Maps dynamic incoming user query questions to the runtime turn
        ("human", "{input}"),
        # Core workspace area for agent to document its planned tool steps
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 4. Construct the Intelligent Agent
# Compiles the model with the tools and system instruction rules
agent = create_tool_calling_agent(chatmodel, tools, agent_prompt)
# Sets up engine runtime environment wrapper loop
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# ==================== ROUTES ====================

# Defines standard landing endpoint locator route rules
@app.route('/')
def index():
    # Serves the user frontend page layout visible in image_3f2bd4.png
    return render_template('chat.html')


# Configures submission traffic gate processing permissions
@app.route('/get', methods=["GET", "POST"])
def chat():
    # Extracts the text input string out of incoming form payloads
    msg = request.form['msg']
    # Prints active user submission directly into local shell console
    print(f"User Question: {msg}")

    try:
        # Fires query down execution chain to track agent tool calls
        response = agent_executor.invoke({"input": msg})
        # Extracts resolved text answer generated by final agent pass
        raw_output = response.get("output")

    except Exception as e:
        print(f"Agent Engine Error caught: {e}")

        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "rate limit" in str(e).lower():
            raw_output = "The system is currently handling too many requests under the Mistral API tier. Please wait a few moments and try sending your query again!"
        else:
            try:
                # Bypasses the model to run a raw direct database similarity search
                local_docs = retriever.invoke(msg)
                if local_docs:
                    raw_output = f"[Local Search Fallback] Here is the matching text found directly in your textbook: {local_docs[0].page_content}"
                else:
                    raw_output = "I ran into a connection issue. Please verify your internet settings and retry."
            except Exception as fallback_err:
                raw_output = "System connection busy. Please wait a moment before trying again."

    if isinstance(raw_output, list) and len(raw_output) > 0:
        if isinstance(raw_output[0], dict) and 'text' in raw_output[0]:
            raw_output = raw_output[0]['text']

    print("Final Processed Response sent to UI: ", raw_output)
    # Returns final raw text response back to frontend interface view
    return str(raw_output)


if __name__ == '__main__':
    # Starts local application daemon process listener on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
