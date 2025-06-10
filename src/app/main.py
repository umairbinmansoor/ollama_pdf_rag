"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""

import streamlit as st
import pandas as pd
import io
import logging
import os
import tempfile
import shutil
import pdfplumber
# import ollama
import warnings
import csv

# Suppress torch warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from streamlit import cache_data
from dotenv import load_dotenv
from utils import *
from typing import List, Tuple, Dict, Any, Optional

# # Load environment variables
# load_dotenv()
# # Get Groq API key from environment variables
# groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key = headers["authorization"]

# Set protobuf environment variable to avoid error messages
# This might cause some issues with latency but it's a tradeoff
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "vectors")

# Streamlit page configuration
st.set_page_config(
    page_title="Multimodal RAG Playground",
    # page_description="A Streamlit app for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info: Response from ollama.list()

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    try:
        # The new response format returns a list of Model objects
        if hasattr(models_info, "models"):
            # Extract model names from the Model objects
            model_names = tuple(model.model for model in models_info.models)
        else:
            # Fallback for any other format
            model_names = tuple()
            
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()


def create_vector_db(file_upload) -> FAISS:#Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        # data = loader.load()
        data = [
            Document(
                page_content=doc.page_content,
                metadata={"source": doc.metadata.get("source", "pdf")}  # Limit metadata
            )
            for doc in loader.load()
        ]

    # Extract images
    images_mapping = extract_images_from_pdf(path, temp_dir)
    # Image to JSON conversion
    json_img_list = JSON_extractor(images_mapping, temp_dir, groq_api_key)

    # Converting image-based JSON documents to LangChain Document objects
    image_documents = [
    Document(page_content=json_data, 
             metadata={"source": "image", "label": label})
    for json_data, label in zip(json_img_list, images_mapping.keys())
    ]
    # Step 2: Combine them
    all_documents = data + image_documents

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents)
    logger.info("Document split into chunks")

    # Updated embeddings configuration with persistent storage
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
        # persist_directory=PERSIST_DIRECTORY,
        # collection_name=f"pdf_{hash(file_upload.name)}"  # Unique collection name per file
    )
    vector_db.save_local("/tmp/faiss_index")
    logger.info("Vector DB created with persistent storage")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db


# def process_question(question: str, vector_db: FAISS, selected_model: str) -> str:
def process_question(question: str, vector_db: FAISS, selected_model: str) -> Dict[str, str]:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    if not isinstance(vector_db, FAISS):
        logger.error(f"Invalid vector_db type: {type(vector_db)}")
        raise ValueError("vector_db must be a FAISS instance")

    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    # Initialize LLM
    # llm = ChatOllama(model=selected_model)
    llm = ChatGroq(
                    model=selected_model,
                    temperature=0.7,
                    api_key=groq_api_key)
    
    # Query prompt template
    # QUERY_PROMPT = PromptTemplate(
    #     input_variables=["question"],
    #     template="""You are an AI language model assistant. Your task is to generate 2
    #     different versions of the given user question to retrieve relevant documents from
    #     a vector database. By generating multiple perspectives on the user question, your
    #     goal is to help the user overcome some of the limitations of the distance-based
    #     similarity search. Provide these alternative questions separated by newlines.
    #     Original question: {question}""",
    # )
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Generate 1 alternative version of the given user question to retrieve relevant documents from a vector database. Provide the alternative question.
                    Original question: {question}"""
    )
    

    # Set up retriever with metadata filtering
    def filter_metadata(docs):
        return [
            Document(
                page_content=doc.page_content,
                metadata={k: v for k, v in doc.metadata.items() if k in ["source", "label"]}
            )
            for doc in docs
        ]
    # Set up retriever
    # retriever = MultiQueryRetriever.from_llm(
    #     vector_db.as_retriever(), 
    #     llm,
    #     prompt=QUERY_PROMPT
    # )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(search_kwargs={"k": 3}),
        llm,
        prompt=QUERY_PROMPT
    ).configurable_fields(transform=filter_metadata)

    # docs = retriever.invoke(question)
    # context = "\n\n".join([doc.page_content for doc in docs])
    
    # Single retrieval call
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    # # Prompt to answer the question based on context
    # template = """Answer the question based ONLY on the following context:
    # {context}
    # Question: {question}
    # """

    # RAG prompt template
    template = """Answer the question based ONLY on the following context, ignoring internal metadata (e.g., IDs, UUIDs):
    {context}
    Question: {question}
    If the question requests a downloadable CSV, format the table data as a valid CSV string with headers and rows, like:
    ```csv
    header1,header2,...
    value1,value2,...
    """
    prompt = ChatPromptTemplate.from_template(template)

    # chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    #     )

    chain = prompt | llm | StrOutputParser()

    # response = chain.invoke({"context": context, "question": question})
    response = ""
    for chunk in chain.stream({"context": context, "question": question}):
        response += chunk
    logger.info("Question processed and response generated")

    is_csv = any(keyword in question.lower() for keyword in ["csv", "download"]) and "table" in question.lower()
    return {
        "response": response,
        "is_csv": is_csv,
        "context": context
        }
    # return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[FAISS]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            # Delete the collection
            vector_db.delete_collection()
            
            # Clear session state
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


@st.cache_resource
def cached_create_vector_db(file_upload, _hash=hash):
    return create_vector_db(file_upload)

@cache_data
def cached_process_question(question: str, _vector_db: FAISS, model: str) -> Dict[str, str]:
    logger.info(f"Caching question: {question} with model: {model}")
    try:
        result = process_question(question, _vector_db, model)
        logger.info(f"Cache hit for question: {question}")
        return result
    except Exception as e:
        logger.error(f"Cache error for question {question}: {e}")
        raise


def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    try:
        st.subheader("🧠 Multimodal RAG playground", divider="gray", anchor=False)

        # Get available models
        available_models = (
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct"
        )

        # Create layout
        col1, col2 = st.columns([1.5, 2])

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "vector_db" not in st.session_state:
            st.session_state["vector_db"] = None

        # Model selection
        if available_models:
            selected_model = col2.selectbox(
                "Pick a model available locally on your system ↓", 
                available_models,
                key="model_select"
            )
            if selected_model not in available_models:
                st.error("Invalid model selected")
                return

        # File upload
        file_upload = col1.file_uploader(
            "Upload a PDF file ↓", 
            type="pdf", 
            accept_multiple_files=False,
            key="pdf_uploader"
        )

        if file_upload:
            # if st.session_state["vector_db"] is None:
            if st.session_state["vector_db"] is None or st.session_state.get("file_upload") != file_upload:
                with st.spinner("Processing uploaded PDF..."):
                    # st.session_state["vector_db"] = cached_create_vector_db(file_upload)
                    file_hash = hash(file_upload.getvalue())
                    st.session_state["vector_db"] = cached_create_vector_db(file_upload, file_hash)
                    st.session_state["file_upload"] = file_upload

                    with pdfplumber.open(file_upload) as pdf:
                        st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]

        # Display PDF pages if available
        if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
            zoom_level = col1.slider(
                "Zoom Level", 
                min_value=100, 
                max_value=1000, 
                value=700, 
                step=50,
                key="zoom_slider"
            )

            with col1:
                with st.container(height=410, border=True):
                    for page_image in st.session_state["pdf_pages"]:
                        st.image(page_image, width=zoom_level)

        # Delete collection button
        delete_collection = col1.button(
            "⚠️ Delete collection", 
            type="secondary",
            key="delete_button"
        )

        if delete_collection:
            # delete_vector_db(st.session_state["vector_db"])
            delete_vector_db(st.session_state["vector_db"])
            st.session_state.clear()  # Clear all session state
            st.session_state["messages"] = []
            st.experimental_rerun()

        # Chat interface
        with col2:
            message_container = st.container(height=500, border=True)
            max_messages = 20
            for message in st.session_state["messages"][-max_messages:]:
                avatar = "🤖" if message["role"] == "assistant" else "😎"
                with message_container.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
                try:
                    st.session_state["messages"].append({"role": "user", "content": prompt})
                    with message_container.chat_message("user", avatar="😎"):
                        st.markdown(prompt)

                    with message_container.chat_message("assistant", avatar="🤖"):
                        with st.spinner(":green[processing...]"):
                            if st.session_state["vector_db"] is not None:
                                result = cached_process_question(
                                    prompt, st.session_state["vector_db"], selected_model
                                )
                                st.markdown(result['response'])
                            else:
                                st.warning("Please upload a PDF file first.")

                    if st.session_state["vector_db"] is not None:
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": result['response']}
                        )

                    if result["is_csv"]:
                        try:
                            csv_string = result["response"].split("```csv\n")[1].split("```")[0]
                            df = pd.read_csv(io.StringIO(csv_string))
                        except (IndexError, pd.errors.ParserError):
                            table_data = result["context"]
                            rows = list(csv.reader(table_data.split("\n")))
                            if rows:
                                headers = rows[0]
                                data = rows[1:] if len(rows) > 1 else []
                                df = pd.DataFrame(data, columns=headers)
                            else:
                                st.error("No table data found")
                                return
                        
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="Download Table as CSV",
                            data=csv_buffer.getvalue(),
                            file_name="table_data.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(e, icon="⛔️")
                    logger.error(f"Error processing prompt: {e}")
            else:
                if st.session_state["vector_db"] is None:
                    st.warning("Upload a PDF file to begin chat...")
    except Exception as e:
        st.error(f"An error occurred: {e}", icon="⛔️")
        logger.error(f"Main loop error: {e}")


if __name__ == "__main__":
    main()