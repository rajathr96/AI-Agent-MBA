import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import nest_asyncio

# Apply nest_asyncio at the very beginning to allow nested event loops
nest_asyncio.apply()

# --- Configuration ---
# Your Google Gemini API key as provided
GOOGLE_API_KEY = "AIzaSyB2jPzfxtfKqsGgRaRTlSV8kQlm49gxUJM"
# LOCAL_DOC_PATH is now taken from user input, removed hardcoded value

# --- Streamlit UI ---
st.set_page_config(page_title="MBA Document AI Agent", layout="wide")

# Custom CSS to change the background to white
st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFF; /* White background */
    }
    .stApp {
        background-color: #FFFFFF; /* Ensures the main app container is also white */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📚 MBA Document AI Agent")
st.markdown("Upload your MBA study materials, and I'll answer your questions based on them!")

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "initial_load_done" not in st.session_state:
    st.session_state.initial_load_done = False
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "user_local_doc_path" not in st.session_state: # New: Store user's input path
    st.session_state.user_local_doc_path = ""


# --- Functions for Document Processing ---
def get_text_chunks(_documents):
    """
    Splits a list of Langchain Document objects into smaller, manageable chunks.
    This helps the RAG model focus on relevant smaller pieces of information.
    """
    print("\n--- Starting: get_text_chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for i, doc in enumerate(_documents):
        print(f"  Splitting document {i+1}/{len(_documents)}: {doc.metadata.get('source', 'Unknown Document')}")
        chunks.extend(text_splitter.split_documents([doc]))
    print(f"  Finished splitting. Total chunks created: {len(chunks)}")
    print("--- Finished: get_text_chunks ---\n")
    return chunks

def create_vector_store(_text_chunks):
    """
    Creates a FAISS vector store from text chunks using Google Gemini embeddings.
    FAISS (Facebook AI Similarity Search) is used for efficient similarity search.
    Embeddings convert text into numerical vectors that capture semantic meaning.
    """
    print("\n--- Starting: create_vector_store (Generating embeddings and building FAISS index) ---")

    embeddings = None
    try:
        print("  Attempting to initialize GoogleGenerativeAIEmbeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        print("  GoogleGenerativeAIEmbeddings initialized successfully.")
    except Exception as e:
        print(f"  ERROR: Failed to initialize GoogleGenerativeAIEmbeddings: {e}")
        st.error(f"Failed to initialize embedding model. Check your API key and internet connection: {e}")
        return None # Return None if initialization fails

    vector_store = None
    if embeddings: # Only proceed if embeddings was initialized
        try:
            print(f"  Generating embeddings for {len(_text_chunks)} text chunks and building FAISS index...")
            vector_store = FAISS.from_documents(_text_chunks, embedding=embeddings)
            print("  FAISS index created successfully.")
        except Exception as e:
            print(f"  ERROR: Failed to generate embeddings or build FAISS index: {e}")
            st.error(f"Failed to generate embeddings or build knowledge base. Check your API key and network: {e}")
            return None # Return None if this step fails

    print("--- Finished: create_vector_store ---\n")
    return vector_store

def get_conversational_chain_instance():
    """
    Defines and returns a new instance of the Retrieval-Augmented Generation (RAG) chain.
    """
    print("\n--- Initializing: Conversational Chain Instance ---")
    prompt_template = """
    You are a helpful and knowledgeable AI assistant specializing in MBA topics.
    Answer the question as thoroughly as possible from the provided context only.
    If the answer is not found in the context, politely state that you cannot provide an answer from the given information.
    Do not try to make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        model,
        chain_type="stuff",
        retriever=st.session_state.vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print("--- Conversational Chain Instance Initialized ---\n")
    return chain

def load_documents_from_local_path(_directory_path):
    """
    Scans the specified local directory (and its subdirectories) to find and load
    PDF and plain text (.txt) files. It avoids reprocessing files already loaded
    in the current session.
    """
    print(f"\n--- Starting: load_documents_from_local_path for '{_directory_path}' ---")
    all_documents = []
    newly_processed_files_count = 0

    if not os.path.isdir(_directory_path):
        st.error(f"The specified directory does not exist: `{_directory_path}`. Please ensure the path is correct.")
        print(f"Error: Directory does not exist: {_directory_path}")
        return [], 0

    for root, _, files in os.walk(_directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Only process if the file hasn't been processed yet in this session
            if file_path not in st.session_state.processed_files:
                print(f"  Processing new file: {file_name}")
                if file_name.endswith(".pdf"):
                    try:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        all_documents.extend(docs)
                        st.session_state.processed_files.add(file_path)
                        newly_processed_files_count += 1
                        print(f"    Loaded PDF: {file_name}")
                    except Exception as e:
                        st.warning(f"Could not load PDF file `{file_name}`: {e}")
                        print(f"    Error loading PDF {file_name}: {e}")
                elif file_name.endswith(".txt"):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        doc = Document(page_content=content, metadata={"source": file_name, "path": file_path})
                        all_documents.append(doc)
                        st.session_state.processed_files.add(file_path)
                        newly_processed_files_count += 1
                        print(f"    Loaded TXT: {file_name}")
                    except Exception as e:
                        st.warning(f"Could not load text file `{file_name}`: {e}")
                        print(f"    Error loading TXT {file_name}: {e}")
            else:
                print(f"  Skipping already processed file: {file_name}")

    print(f"--- Finished: load_documents_from_local_path. Loaded {len(all_documents)} documents. ---\n")
    return all_documents, newly_processed_files_count


# --- User Input for Document Path ---
st.header("1. Specify Your MBA Documents Folder")
# Update the default value with your last used path for convenience
input_path = st.text_input(
    "Enter the full local path to your MBA documents folder:",
    value=st.session_state.user_local_doc_path or "/Users/rajathr/Downloads/OneDrive_1_11-05-2025 2/Term 1/Organisation Design", # Default if not set
    placeholder="e.g., /Users/yourname/Documents/MBA_Materials",
    key="path_input_field" # Unique key for the widget
)

# Only proceed if the input path is not empty and has changed
if input_path and input_path != st.session_state.user_local_doc_path:
    st.session_state.user_local_doc_path = input_path
    st.session_state.initial_load_done = False # Reset flag to re-trigger load for new path
    st.session_state.vector_store = None # Clear old vector store
    st.session_state.processed_files = set() # Clear processed files
    st.session_state.qa_chain = None # Clear old QA chain
    st.session_state.last_response = None # Clear old response
    st.rerun() # Rerun the script to apply the new path and start processing

# --- Initial Document Loading and Processing on App Start / Path Change ---
# This block now runs only if a path is provided AND initial_load_done is False
if st.session_state.user_local_doc_path and not st.session_state.initial_load_done:
    st.header("2. Building Your Knowledge Base")
    with st.spinner(f"Scanning `{st.session_state.user_local_doc_path}` for MBA documents. This might take a moment if the directory is large..."):
        documents_from_path, num_newly_processed = load_documents_from_local_path(st.session_state.user_local_doc_path)

        if documents_from_path:
            print("\n--- Starting overall document processing (chunking and vector store creation) ---")
            raw_text_chunks = get_text_chunks(documents_from_path)
            if st.session_state.vector_store: # This branch is unlikely now, as we clear it on path change
                print("  Adding new documents to existing vector store.")
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
                st.session_state.vector_store.add_documents(raw_text_chunks, embeddings=embeddings)
            else:
                print("  Creating a new vector store.")
                st.session_state.vector_store = create_vector_store(raw_text_chunks)
            
            if st.session_state.vector_store:
                if st.session_state.qa_chain is None:
                    st.session_state.qa_chain = get_conversational_chain_instance()

                st.success(f"Loaded and processed {num_newly_processed} new documents from `{st.session_state.user_local_doc_path}`.")
                st.info(f"Total unique documents in knowledge base: {len(st.session_state.processed_files)}.")
                st.session_state.initial_load_done = True
                print("--- Finished overall document processing ---\n")
            else:
                st.error("Failed to set up the knowledge base. Please check the console for errors and your API key.")
                st.session_state.initial_load_done = True
        else:
            st.warning(f"No PDF or TXT documents found or processed in `{st.session_state.user_local_doc_path}`. Please check the path and file types.")
            st.session_state.initial_load_done = True
            print(f"Warning: No documents found in {st.session_state.user_local_doc_path}")


# --- Question Answering Interface ---
if st.session_state.vector_store and st.session_state.qa_chain:
    st.header("3. Ask a Question about your MBA Documents")
    user_question = st.text_area("Enter your question here:", placeholder="e.g., What are the key principles of Porter's Five Forces?", height=100)

    if user_question:
        with st.spinner("Finding an answer..."):
            response = st.session_state.qa_chain.invoke({"query": user_question})
            st.session_state.last_response = response # Store the response in session state

    if st.session_state.last_response:
        answer_tab, sources_tab = st.tabs(["Answer", "Sources"])

        with answer_tab:
            st.subheader("Answer:")
            st.write(st.session_state.last_response["result"])

        with sources_tab:
            st.subheader("Sources:")
            if st.session_state.last_response.get("source_documents"):
                for i, doc in enumerate(st.session_state.last_response["source_documents"]):
                    source_info = doc.metadata.get('path', doc.metadata.get('source', 'Unknown Source'))
                    st.write(f"- Source {i+1}: `{source_info}` (Page: {doc.metadata.get('page', 'N/A')})")
                    with st.expander(f"View content from Source {i+1}"):
                        st.text(doc.page_content[:500] + "...")
            else:
                st.info("No source documents were found for this answer.")
elif st.session_state.user_local_doc_path: # If path is set but vector store/chain failed
    st.info("Knowledge base setup is in progress or encountered an issue. Please check console for details.")
else: # If no path is provided yet
    st.info("Please enter the path to your MBA documents folder above to get started.")


st.markdown("---")
st.caption("Powered by Streamlit, LangChain, and Google Gemini API.")
