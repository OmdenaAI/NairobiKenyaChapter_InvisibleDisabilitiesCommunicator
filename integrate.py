import streamlit as st
from pathlib import Path
import os
import asyncio
import pickle
import warnings
import pandas as pd
from dotenv import load_dotenv

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_groq import ChatGroq

# import gettext

# from googletrans import Translator
from deep_translator import GoogleTranslator

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Set API keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# File paths
CSV_FILE_PATH = "temp_matatu_routes.csv"
FAISS_INDEX_PATH = "faiss_index.pkl"

# Language sidebar selection
language = st.sidebar.selectbox("Select Language", ["English", "Swahili"])

# Configure translations
translator = GoogleTranslator(
    source="auto", target="sw" if language == "Swahili" else "en"
)


# Cache LLM to prevent reloading on every query
@st.cache_resource
def load_llm():
    return ChatGroq(model="deepseek-r1-distill-llama-70b")


# Cache embeddings and vector store for fast retrieval
@st.cache_resource
def load_vector_store():
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            with open(FAISS_INDEX_PATH, "rb") as f:
                return pickle.load(f).as_retriever()
        except Exception:
            st.warning("FAISS index is corrupted. Rebuilding...")

    # Load CSV data
    loader = CSVLoader(file_path=Path(CSV_FILE_PATH))
    docs = loader.load_and_split()

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings()
    index = faiss.IndexFlatL2(len(embeddings.embed_query(" ")))

    # Create FAISS vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(docs)

    # Save FAISS index
    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(vector_store, f)

    return vector_store.as_retriever()


llm = load_llm()
retriever = load_vector_store()

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# Async function to handle retrieval in the background
async def async_retrieve_answer(query):
    return await asyncio.to_thread(rag_chain.invoke, {"input": query})


def main():
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        /* Remove default padding and margin */
        html, body, .stApp {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
        }

        /* Full-width background gradient */
        .stApp {
            background: linear-gradient(to right, #74ebd5, #acb6e5);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        /* Full-width output window */
        .output-window {
            background-color: #ffffff;
            padding: 20px;
            border: 1px solid #cccccc;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            width: 90%; /* Use 90% of the screen width */
            height: 300px; /* Increased height */
            overflow-y: auto;
            margin: 20px auto; /* Center alignment */
            text-align: left;
        }

        /* Full-width input section */
        .input-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 20px;
            width: 90%; /* Use 90% of the screen width */
            margin-left: auto;
            margin-right: auto;
        }

        /* Input box styling */
        .input-box {
            background-color: #ffffff;
            padding: 10px;
            border: 1px solid #cccccc;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px; /* Space between input box and button */
        }

        /* Submit button styling */
        .submit-button {
            background-color: #00698f;
            color: #ffffff;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 200px; /* Adjust the width as needed */
        }

        .submit-button:hover {
            background-color: #003d5c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title Section
    st.markdown(
        "<h1 style='text-align: center; font-size: 50px; color: #003d5c;'>Invisible Disability Assistant</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 20px; color: #444;'>Nairobi Disability Support Platform</p>",
        unsafe_allow_html=True,
    )

    # Output Window Placeholder
    output_window = st.empty()

    # User Input
    query = st.text_input("", placeholder="Type your query here", key="input-field")

    # Submit Button
    if st.button("Submit", key="submit-button"):
        if query:
            try:
                response = asyncio.run(async_retrieve_answer(query))
                # output_window.write(response["answer"])
                translated_response = response["answer"]

                # Translate AI response is language is swahili
                if language == "Swahili":
                    translated_response = translator.translate(
                        response["answer"], dest="sw"
                    ).text

                st.write(translated_response)
            except Exception as e:
                st.write(f"Error: {e}")
        else:
            st.write("Please enter a query!")


if __name__ == "__main__":
    main()
