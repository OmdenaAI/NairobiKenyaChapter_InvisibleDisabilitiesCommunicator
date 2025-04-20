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

import gettext
from deep_translator import (
    GoogleTranslator,
)  # Replaced googletrans with deep-translator

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

# Configure gettext for UI translations
locales_dir = "locales"
translation = gettext.translation(
    "messages",
    localedir=locales_dir,
    languages=["sw" if language == "Swahili" else "en"],
    fallback=True,
)
translation.install()
_ = translation.gettext  # Shortcut for gettext translations

# Configure AI response translation
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
            st.warning(_("FAISS index is corrupted. Rebuilding..."))

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
system_prompt = _(
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If the context does not contain relevant "
    "information, respond with 'I do not know'. Do not add "
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
    # Title Section with gettext translations
    st.markdown(
        f"<h1 style='text-align: center; font-size: 50px; color: #003d5c;'>{_('Invisible Disability Assistant')}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align: center; font-size: 20px; color: #444;'>{_('Nairobi Disability Support Platform')}</p>",
        unsafe_allow_html=True,
    )

    # User Input Section
    query = st.text_input("", placeholder=_("Type your query here"), key="input-field")

    # Submit Button
    if st.button(_("Submit"), key="submit-button"):
        if query:
            try:
                # Use asyncio.create_task() instead of asyncio.run()
                response_task = asyncio.create_task(async_retrieve_answer(query))
                response = asyncio.run(response_task)

                translated_response = response["answer"]

                # Translate AI response if Swahili is selected
                if language == "Swahili":
                    translated_response = translator.translate(response["answer"])

                st.write(translated_response)

            except Exception as e:
                st.write(f"Error: {e}")
        else:
            st.write(_("Please enter a query!"))


if __name__ == "__main__":
    main()
