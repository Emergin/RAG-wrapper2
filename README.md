# RAG Wrapper 📚

A Retrieval-Augmented Generation (RAG) wrapper that processes user documents, embeds them using TF-IDF, and provides factual answers using GPT models.

## Features

- **Document Upload**: Support for PDF, Markdown, TXT, DOCX, and CSV files
- **TF-IDF Embeddings**: Lightweight document embeddings without external API dependencies
- **Vector Storage**: Efficient similarity search using FAISS
- **GPT Integration**: Uses OpenAI's GPT models for high-quality generation
- **Explainability Dashboard**: Understand why specific documents were retrieved
- **User-friendly Interface**: Built with Streamlit for ease of use

## Project Structure

```
rag-wrapper/
├── requirements.txt
├── app.py                  # Main Streamlit application
├── utils/
│   ├── __init__.py
│   ├── document_loader.py  # Document loading utilities
│   ├── embeddings.py       # TF-IDF embedding generation
│   ├── vector_store.py     # Vector database operations
│   └── rag_pipeline.py     # RAG implementation with GPT
├── data/                   # Directory to store uploaded files
└── README.md               # Project documentation
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/rag-wrapper.git
   cd rag-wrapper
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root directory
   - Add your API key to the file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Alternatively, you can provide the API key in the web interface

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your browser and navigate to the displayed URL (typically http://localhost:8501)

3. Upload your documents using the sidebar

4. Process the documents by clicking the "Process Files" button

5. Ask questions about your documents in the chat interface

## How It Works

1. **Document Processing**:
   - When you upload documents, they are loaded and split into chunks
   - Each chunk is processed using a TF-IDF vectorizer to create embeddings
   - The embeddings are stored in a FAISS vector index for efficient retrieval

2. **Retrieval**:
   - When you ask a question, the question is converted to a TF-IDF embedding
   - The system performs a similarity search to find the most relevant document chunks
   - The top-k most similar chunks are retrieved

3. **Generation**:
   - The retrieved document chunks are sent to the GPT model along with your question
   - The GPT model generates an answer based on the provided context

4. **Explainability**:
   - When the "Show explainability analysis" option is enabled, you can see why specific documents were retrieved
   - The system displays the most important terms in your query and in each retrieved document

## Requirements

- Python 3.8+
- OpenAI API key
- 4GB+ RAM (recommended)

## Limitations

- TF-IDF embeddings are less semantically rich than neural embeddings like those from OpenAI or Hugging Face
- Performance may degrade with very large document collections
- The quality of answers depends on the quality of the GPT model and the retrieved documents

## Future Improvements

- Add support for neural embeddings
- Implement document caching for faster processing
- Add authentication for multi-user scenarios
- Improve document chunking strategies
- Add support for more file formats