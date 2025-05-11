import os
import PyPDF2
import markdown
import docx
import pandas as pd
import re
from typing import Dict, List, Tuple

class DocumentLoader:
    """Utility class for loading different document types"""
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Load and extract text from a PDF file"""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def load_markdown(file_path: str) -> str:
        """Load and extract text from a Markdown file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Convert markdown to plain text (simplified)
            html = markdown.markdown(text)
            text = re.sub('<[^<]+?>', '', html)  # Simple HTML tag removal
        return text
    
    @staticmethod
    def load_text(file_path: str) -> str:
        """Load a plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    @staticmethod
    def load_docx(file_path: str) -> str:
        """Load and extract text from a Word document"""
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    @staticmethod
    def load_csv(file_path: str) -> str:
        """Load and convert CSV to text"""
        df = pd.read_csv(file_path)
        return df.to_string()
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            # If we're not at the end of the document, try to break at a paragraph or sentence
            if end < text_len:
                # Try to find paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Try to find sentence break
                    sentence_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('? ', start, end),
                        text.rfind('! ', start, end)
                    )
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 2
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    @classmethod
    def load_document(cls, file_path: str) -> Tuple[str, List[str]]:
        """Load a document based on its file extension and chunk it"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pdf':
            text = cls.load_pdf(file_path)
        elif ext == '.md':
            text = cls.load_markdown(file_path)
        elif ext == '.txt':
            text = cls.load_text(file_path)
        elif ext == '.docx':
            text = cls.load_docx(file_path)
        elif ext == '.csv':
            text = cls.load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        chunks = cls.chunk_text(text)
        return text, chunks