from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss # just for pipreqs to detect the dependency
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import logging

class FAISSTextbookIndexer:
    def __init__(self, textbook_path: str, index_path: str = "./indexes"):
        """
        Initialize the textbook indexer
        
        Args:
            textbook_path: Path to the PDF textbook
            index_path: Path to store the FAISS index
        """
        self.textbook_path = textbook_path
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.embeddings = OpenAIEmbeddings()
        
    def create_index(self) -> FAISS:
        """Create a FAISS index from the textbook"""
        # Load PDF
        loader = PyPDFLoader(self.textbook_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create and save index
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        index_name = Path(self.textbook_path).stem
        vectorstore.save_local(str(self.index_path / index_name))
        
        return vectorstore
    
    def load_index(self, index_name: str) -> FAISS:
        """Load an existing FAISS index"""
        return FAISS.load_local(
            str(self.index_path / index_name), 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def get_relevant_content(self, query: str, index_name: str, k: int = 3) -> list:
        """
        Get relevant content from the textbook based on a query
        
        Args:
            query: Query text to search for
            index_name: Name of the index to search
            k: Number of relevant chunks to return
            
        Returns:
            List of relevant text chunks
        """
        try:
            vectorstore = self.load_index(index_name)
            results = vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            logging.error(f"Error retrieving content: {str(e)}")
            return []