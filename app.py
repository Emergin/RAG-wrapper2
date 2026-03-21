import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import numpy as np
from typing import List, Dict, Any
import time

from utils.document_loader import DocumentLoader
from utils.embeddings import TFIDFEmbedder
from utils.vector_store import VectorStore
from utils.rag_pipeline import RAGPipeline

# Set page configuration
st.set_page_config(
    page_title="RAG Wrapper",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('data/uploads', exist_ok=True)

# Helper functions
def initialize_pipeline():
    """Initialize the RAG pipeline components"""
    if not st.session_state.embedder:
        st.session_state.embedder = TFIDFEmbedder(max_features=10000)
    if not st.session_state.vector_store:
        st.session_state.vector_store = VectorStore()
    if not st.session_state.rag_pipeline and st.session_state.embedder and st.session_state.vector_store:
        st.session_state.rag_pipeline = RAGPipeline(
            st.session_state.vector_store,
            st.session_state.embedder
        )

def process_uploaded_file(uploaded_file):
    """Process an uploaded file and add it to the vector store"""
    # Save uploaded file to disk temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load and chunk the document
        text, chunks = DocumentLoader.load_document(tmp_path)
        
        # Check if we need to fit the embedder
        first_upload = len(st.session_state.uploaded_files) == 0
        
        # Generate embeddings
        if first_upload:
            embeddings = st.session_state.embedder.fit_transform(chunks)
        else:
            embeddings = st.session_state.embedder.transform(chunks)
        
        # Add to vector store
        st.session_state.vector_store.add_documents(
            embeddings=embeddings,
            chunks=chunks,
            source=uploaded_file.name
        )
        
        # Add to session state
        st.session_state.uploaded_files.append({
            'name': uploaded_file.name,
            'chunks': len(chunks),
            'size': uploaded_file.size
        })
        
        return True
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def display_stats():
    """Display statistics about the uploaded files and vector store"""
    if st.session_state.vector_store:
        stats = st.session_state.vector_store.get_stats()
        
        st.subheader("📊 Document Stats")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Documents", stats["document_count"])
        col2.metric("Unique Files", stats["unique_sources"])
        col3.metric("Embedding Dimension", stats["embedding_dimension"] if "embedding_dimension" in stats else "N/A")
        
        # Display uploaded files table
        if st.session_state.uploaded_files:
            df = pd.DataFrame(st.session_state.uploaded_files)
            st.dataframe(df, use_container_width=True)

def render_chat_ui():
    """Render the chat interface"""
    st.subheader("💬 Ask Questions About Your Documents")
    
    # Query input
    query = st.text_input("Ask a question:", key="query_input")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=3)
    with col2:
        model = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    with col3:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    with col4:
        show_analysis = st.checkbox("Show explainability analysis", value=st.session_state.show_analysis)
        st.session_state.show_analysis = show_analysis
    
    submit = st.button("Submit")
    
    if submit and query:
        with st.spinner("Generating answer..."):
            # Get answer using RAG pipeline
            response = st.session_state.rag_pipeline.query(
                query=query,
                k=k,
                model=model,
                temperature=temperature
            )
            
            # Add to chat history
            st.session_state.chat_history.append({
                "query": query,
                "response": response,
                "timestamp": time.time()
            })
            
            # Get retrieval analysis if requested
            if show_analysis:
                analysis = st.session_state.rag_pipeline.analyze_retrieval(query, k=k)
                response["analysis"] = analysis
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"### Q: {chat['query']}")
            st.markdown(f"### A: {chat['response']['answer']}")
            
            with st.expander("View sources"):
                for j, (source, chunk) in enumerate(zip(chat['response']['sources'], chat['response']['chunks'])):
                    st.markdown(f"**Source {j+1}**: {source}")
                    st.text(chunk)
            
            if 'analysis' in chat['response'] and st.session_state.show_analysis:
                with st.expander("View analysis"):
                    display_analysis(chat['response']['analysis'])
            
            st.markdown("---")

def display_analysis(analysis):
    """Display explainability analysis"""
    st.subheader("Query Analysis")
    
    # Display query terms
    query_terms_df = pd.DataFrame(analysis['query_terms'], columns=["Term", "Score"])
    
    fig = px.bar(
        query_terms_df, 
        x="Term", 
        y="Score", 
        title="Most Important Terms in Query",
        labels={"Term": "Term", "Score": "TF-IDF Score"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display document terms
    st.subheader("Retrieved Documents Analysis")
    
    for i, doc_analysis in enumerate(analysis['doc_terms']):
        with st.expander(f"Document {i+1}: {doc_analysis['source']} (Score: {doc_analysis['score']:.4f})"):
            doc_terms_df = pd.DataFrame(doc_analysis['terms'], columns=["Term", "Score"])
            
            fig = px.bar(
                doc_terms_df, 
                x="Term", 
                y="Score", 
                title=f"Most Important Terms in Document {i+1}",
                labels={"Term": "Term", "Score": "TF-IDF Score"}
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    # Set title and description
    st.title("📚 RAG Wrapper")
    st.markdown("""
    This application allows you to upload documents and ask questions about them.
    The system uses TF-IDF to create document embeddings and a Retrieval-Augmented Generation (RAG) pipeline with GPT to provide accurate answers based on your documents.
    """)
    
    # Initialize pipeline
    initialize_pipeline()
    
    # Sidebar
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents", 
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "docx", "csv"]
        )
        
        if uploaded_files:
            process_button = st.button("Process Files")
            if process_button:
                with st.spinner("Processing files..."):
                    success_count = 0
                    for uploaded_file in uploaded_files:
                        if process_uploaded_file(uploaded_file):
                            success_count += 1
                    
                    if success_count > 0:
                        st.success(f"Successfully processed {success_count} files!")
                    else:
                        st.error("Failed to process files.")
        
        # Reset button
        if st.session_state.vector_store and len(st.session_state.uploaded_files) > 0:
            if st.button("Reset System"):
                st.session_state.vector_store = None
                st.session_state.embedder = None
                st.session_state.rag_pipeline = None
                st.session_state.uploaded_files = []
                st.session_state.chat_history = []
                st.session_state.show_analysis = False
                st.success("System reset successfully!")
                initialize_pipeline()
        
        # OpenAI API Key input
        st.subheader("API Configuration")
        api_key = st.text_input("Gemini API Key", type="password")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.set_api_key(api_key)
            st.success("API Key set!")
        
        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This RAG (Retrieval-Augmented Generation) wrapper uses:
        - TF-IDF for document embeddings
        - FAISS for efficient similarity search
        - OpenAI GPT models for generation
        - Streamlit for the user interface
        """)
    
    # Main content
    tab1, tab2 = st.tabs(["Chat", "Document Stats"])
    
    with tab1:
        # Display chat interface only if documents are uploaded
        if st.session_state.vector_store and len(st.session_state.uploaded_files) > 0:
            render_chat_ui()
        else:
            st.info("Please upload and process documents to start asking questions.")
    
    with tab2:
        # Display statistics
        if st.session_state.vector_store and len(st.session_state.uploaded_files) > 0:
            display_stats()
        else:
            st.info("No documents uploaded yet.")

if __name__ == "__main__":
    main()