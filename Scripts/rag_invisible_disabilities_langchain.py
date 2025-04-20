# **Utilizing Retrieval Augmented Generation for Nairobi Invisible Disabilities Classifier.**

# Import libraries
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path
import os
from dotenv import load_dotenv

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_groq import ChatGroq 
from langchain_community.embeddings import HuggingFaceEmbeddings 

from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = ChatGroq(model="mixtral-8x7b-32768")

# Get the path to the parent directory
parent_folder = Path(__file__).parent.parent

# Define the path to the "data" folder
data_folder = parent_folder / "Data"

# Find the Matatu_Routes CSV file in the "Data" folder
file_path = next(data_folder.glob("Matatu_Routes.csv"), None)

# Load and split csv file documents
loader = CSVLoader(file_path=file_path)
docs = loader.load_and_split()

# Initiate faiss vector store and huggingface embedding
embeddings = HuggingFaceEmbeddings()
index = faiss.IndexFlatL2(len(HuggingFaceEmbeddings().embed_query(" ")))
vector_store = FAISS(
    embedding_function=HuggingFaceEmbeddings(),
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store.add_documents(documents=docs)

retriever = vector_store.as_retriever()

# Set up system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If the context does not contain relevant "
    "information, respond with 'I do not know'.  Do not add "
    "any extra information beyond what is in the retrieved "
    "context. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),

])

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Interactive Terminal Chatbot
print("\n🚐 Welcome to the Matatu Route Chatbot! 🚐")
print("Ask a question about Matatu routes in Nairobi (type 'exit' to quit):\n")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        response = rag_chain.invoke({"input": user_input})
        print(f"Bot: {response['answer']}\n")

    except Exception as e:
        print(f"Error: {e}")