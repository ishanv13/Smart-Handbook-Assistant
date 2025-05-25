# Advanced RAG System for College Handbook Q&A

# Import necessary libraries
import os
import re
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import time

# Document processing
import PyPDF2
from docx import Document

# Embedding and search
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
import faiss

# Google Gemini
import google.generativeai as genai
from google.api_core import retry

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangChainDocument
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Utilities
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Configuration settings for the RAG system."""
    
    def __init__(self):
        # API Keys
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyApLv1N5SWMetiowWfTSFTjrGVZWwmPRPI")
        
        # Model settings
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient
        self.LLM_MODEL = "models/gemini-1.5-pro-latest"  # Using the latest stable version
        self.FALLBACK_MODEL = "models/gemini-1.5-flash-latest"  # Fallback to flash model
        self.API_VERSION = "v1"  # Using v1 API
        
        # Document processing
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.TOP_K_RESULTS = 5
        
        # Search settings
        self.HYBRID_SEARCH_ALPHA = 0.7  # Weight for dense search
        
        # File paths
        self.DEFAULT_HANDBOOK_PATH = os.path.normpath(
            "C:/Users/ishaa/Desktop/Gen AI/Individual Assignment/Student Handbook (Master's).pdf"
        )
        
        # Supported file formats
        self.SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt'}
        
        # Initialize Google API
        self._init_google_api()
    
    def _init_google_api(self):
        """Initialize Google API with retry settings."""
        try:
            # Configure the API with explicit API key
            genai.configure(
                api_key=self.GOOGLE_API_KEY,
                transport='rest'  # Use REST transport instead of gRPC
            )
            
            # List available models
            try:
                available_models = [model.name for model in genai.list_models()]
                logger.info(f"Available models: {available_models}")
            except Exception as e:
                logger.error(f"Failed to list models: {str(e)}")
                available_models = []
            
            # Try to find a suitable model with updated priorities
            model_priority = [
                "models/gemini-1.5-pro-latest",    # Latest stable version
                "models/gemini-1.5-pro",           # Stable version
                "models/gemini-1.5-flash-latest",  # Latest flash version
                "models/gemini-1.5-flash",         # Flash version
                "models/gemini-1.5-flash-8b-latest" # Lightweight version
            ]
            
            # Filter out deprecated models
            available_models = [m for m in available_models if not any(dep in m.lower() for dep in ['vision', '1.0', '2.0'])]
            
            for model_name in model_priority:
                if model_name in available_models:
                    self.LLM_MODEL = model_name
                    logger.info(f"Selected model: {model_name}")
                    break
            else:
                # If no preferred model is available, try to find any working model
                for model in available_models:
                    if 'gemini' in model.lower() and 'flash' in model.lower():
                        self.LLM_MODEL = model
                        logger.info(f"Selected fallback model: {model}")
                        break
                else:
                    logger.warning("No suitable models found, using default: models/gemini-1.5-pro-latest")
                    self.LLM_MODEL = "models/gemini-1.5-pro-latest"
            
            # Test the model with exponential backoff
            max_retries = 3
            base_delay = 2  # Base delay in seconds
            
            for attempt in range(max_retries):
                try:
                    model = genai.GenerativeModel(
                        self.LLM_MODEL,
                        generation_config={
                            "temperature": 0,
                            "top_p": 1,
                            "top_k": 1,
                            "max_output_tokens": 2048,
                        }
                    )
                    response = model.generate_content("Test connection")
                    logger.info(f"Successfully tested model: {self.LLM_MODEL}")
                    return  # Success, exit the function
                    
                except Exception as model_error:
                    error_str = str(model_error).lower()
                    
                    # Handle different types of errors
                    if "quota" in error_str or "429" in error_str:
                        # If we hit rate limits, try the fallback model
                        if self.LLM_MODEL != self.FALLBACK_MODEL:
                            logger.warning(f"Rate limit hit with {self.LLM_MODEL}, trying fallback model...")
                            self.LLM_MODEL = self.FALLBACK_MODEL
                            continue
                    elif "deprecated" in error_str:
                        # If model is deprecated, try next model in priority
                        logger.warning(f"Model {self.LLM_MODEL} is deprecated, trying next model...")
                        continue
                    
                    if attempt == max_retries - 1:
                        logger.error(f"Model test failed after {max_retries} attempts: {str(model_error)}")
                        raise
                    
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Model test attempt {attempt + 1} failed, retrying in {delay} seconds...")
                    time.sleep(delay)
            
        except Exception as e:
            logger.error(f"Failed to configure Google API: {str(e)}")
            raise

# Initialize configuration
config = Config()

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str

class DocumentProcessor:
    """Handles document loading and chunking."""
    
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> str:
        """Load text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            if not text.strip():
                raise ValueError("PDF file appears to be empty or contains no extractable text")
            return text
        except PyPDF2.PdfReadError:
            raise ValueError(f"Error reading PDF file: {file_path} - File may be corrupted or password protected")
        except Exception as e:
            raise Exception(f"Error processing PDF file: {str(e)}")
    
    def load_docx(self, file_path: str) -> str:
        """Load text from DOCX file."""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            if not text.strip():
                raise ValueError("DOCX file appears to be empty or contains no text")
            return text
        except Exception as e:
            raise Exception(f"Error processing DOCX file: {str(e)}")
    
    def load_txt(self, file_path: str) -> str:
        """Load text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            if not text.strip():
                raise ValueError("TXT file appears to be empty")
            return text
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                if not text.strip():
                    raise ValueError("TXT file appears to be empty")
                return text
            except Exception as e:
                raise Exception(f"Error reading TXT file with alternative encoding: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing TXT file: {str(e)}")
    
    def load_document(self, file_path: str) -> str:
        """Load document based on file extension."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.suffix.lower() not in config.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats are: {', '.join(config.SUPPORTED_FORMATS)}"
            )
        
        try:
            if file_path.suffix.lower() == '.pdf':
                return self.load_pdf(str(file_path))
            elif file_path.suffix.lower() == '.docx':
                return self.load_docx(str(file_path))
            elif file_path.suffix.lower() == '.txt':
                return self.load_txt(str(file_path))
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def chunk_document(self, text: str, source: str) -> List[DocumentChunk]:
        """Split document into chunks."""
        chunks = self.text_splitter.split_text(text)
        document_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source}_chunk_{i}"
            metadata = {
                "source": source,
                "chunk_index": i,
                "chunk_size": len(chunk)
            }
            document_chunks.append(DocumentChunk(chunk, metadata, chunk_id))
        
        return document_chunks

class AdvancedEmbeddingModel:
    """Advanced embedding model with multiple strategies."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Loaded embedding model: {model_name}")
    
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode list of texts into embeddings."""
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=show_progress,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text into embedding."""
        return self.model.encode([text], normalize_embeddings=True)[0]

class HybridSearchEngine:
    """Combines dense (semantic) and sparse (keyword) search."""
    
    def __init__(self, embedding_model: AdvancedEmbeddingModel):
        self.embedding_model = embedding_model
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = None
        self.bm25 = None
        self.faiss_index = None
        
    def index_documents(self, chunks: List[DocumentChunk]):
        """Index documents for both dense and sparse search."""
        self.chunks = chunks
        texts = [chunk.content for chunk in chunks]
        
        print("Creating dense embeddings...")
        # Dense embeddings for semantic search
        self.embeddings = self.embedding_model.encode_texts(texts)
        
        # FAISS index for fast similarity search
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product for normalized vectors
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        print("Creating sparse index...")
        # Sparse search using BM25
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        print(f"Indexed {len(chunks)} document chunks successfully!")
    
    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Perform dense semantic search."""
        query_embedding = self.embedding_model.encode_single(query)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Perform sparse keyword search using BM25."""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include relevant results
                results.append((self.chunks[idx], scores[idx]))
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.7) -> List[DocumentChunk]:
        """Combine dense and sparse search results.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for dense search (1-alpha for sparse)
        """
        # Get results from both methods
        dense_results = self.dense_search(query, top_k * 2)
        sparse_results = self.sparse_search(query, top_k * 2)
        
        # Normalize scores
        dense_scores = {chunk.chunk_id: score for chunk, score in dense_results}
        sparse_scores = {chunk.chunk_id: score for chunk, score in sparse_results}
        
        # Normalize sparse scores to [0, 1] range
        if sparse_scores:
            max_sparse = max(sparse_scores.values())
            if max_sparse > 0:
                sparse_scores = {k: v/max_sparse for k, v in sparse_scores.items()}
        
        # Combine scores
        all_chunk_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined_scores = {}
        
        for chunk_id in all_chunk_ids:
            dense_score = dense_scores.get(chunk_id, 0)
            sparse_score = sparse_scores.get(chunk_id, 0)
            combined_scores[chunk_id] = alpha * dense_score + (1 - alpha) * sparse_score
        
        # Sort by combined score and return top k chunks
        sorted_chunk_ids = sorted(combined_scores.keys(), 
                                key=lambda x: combined_scores[x], 
                                reverse=True)[:top_k]
        
        # Find corresponding chunks
        chunk_dict = {chunk.chunk_id: chunk for chunk in self.chunks}
        return [chunk_dict[chunk_id] for chunk_id in sorted_chunk_ids if chunk_id in chunk_dict]

class QueryEnhancer:
    """Enhances user queries using LLM for better retrieval."""
    
    def __init__(self, llm_model: str = None):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=llm_model or config.LLM_MODEL,
                temperature=0,
                google_api_key=config.GOOGLE_API_KEY,
                convert_system_message_to_human=True,
                model_kwargs={
                    "generation_config": {
                        "temperature": 0,
                        "top_p": 1,
                        "top_k": 1,
                        "max_output_tokens": 2048,
                    }
                }
            )
            logger.info(f"QueryEnhancer initialized with model: {llm_model or config.LLM_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize QueryEnhancer: {str(e)}")
            raise
        
        # Query expansion prompt
        self.expansion_template = PromptTemplate(
            input_variables=["query"],
            template="""
You are helping to improve a search query for a college handbook Q&A system.

Original query: {query}

Please provide:
1. An improved, more specific version of the query
2. 3-5 related keywords or phrases that might appear in relevant documents
3. Alternative ways to phrase this question

Format your response as:
IMPROVED_QUERY: [improved query]
KEYWORDS: [keyword1, keyword2, keyword3, ...]
ALTERNATIVES: [alt1 | alt2 | alt3]
"""
        )
        
        self.expansion_chain = LLMChain(
            llm=self.llm,
            prompt=self.expansion_template
        )
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """Enhance query with LLM-generated improvements."""
        try:
            response = self.expansion_chain.run(query=query)
            
            # Parse response
            enhanced = {
                "original_query": query,
                "improved_query": query,  # fallback
                "keywords": [],
                "alternatives": [query]  # fallback
            }
            
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('IMPROVED_QUERY:'):
                    enhanced["improved_query"] = line.replace('IMPROVED_QUERY:', '').strip()
                elif line.startswith('KEYWORDS:'):
                    keywords_str = line.replace('KEYWORDS:', '').strip()
                    enhanced["keywords"] = [k.strip() for k in keywords_str.split(',')]
                elif line.startswith('ALTERNATIVES:'):
                    alts_str = line.replace('ALTERNATIVES:', '').strip()
                    enhanced["alternatives"] = [a.strip() for a in alts_str.split('|')]
            
            return enhanced
            
        except Exception as e:
            print(f"Query enhancement failed: {e}")
            return {
                "original_query": query,
                "improved_query": query,
                "keywords": [],
                "alternatives": [query]
            }
    
    def create_expanded_query(self, enhanced: Dict[str, Any]) -> str:
        """Create expanded query string combining all enhancements."""
        parts = [enhanced["improved_query"]]
        
        if enhanced["keywords"]:
            parts.extend(enhanced["keywords"])
        
        return " ".join(parts)

class StepBackPrompter:
    """Implements step-back prompting for better reasoning."""
    
    def __init__(self, llm_model: str = None):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=llm_model or config.LLM_MODEL,
                temperature=0,
                google_api_key=config.GOOGLE_API_KEY,
                convert_system_message_to_human=True,
                model_kwargs={
                    "generation_config": {
                        "temperature": 0,
                        "top_p": 1,
                        "top_k": 1,
                        "max_output_tokens": 2048,
                    }
                }
            )
            logger.info(f"StepBackPrompter initialized with model: {llm_model or config.LLM_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize StepBackPrompter: {str(e)}")
            raise
        
        # Step-back question generation
        self.stepback_template = PromptTemplate(
            input_variables=["question"],
            template="""
You are an expert at asking step-back questions. Given a specific question about college policies or procedures, 
generate a broader, more general question that would help understand the underlying concepts.

Original question: {question}

Generate a step-back question that asks about the general concept, policy category, or broader topic.

Step-back question:"""
        )
        
        self.stepback_chain = LLMChain(
            llm=self.llm,
            prompt=self.stepback_template
        )
        
        # Answer synthesis template
        self.synthesis_template = PromptTemplate(
            input_variables=["original_question", "stepback_question", "stepback_context", "specific_context"],
            template="""
You are answering a student's question about college policies using the college handbook.
Also explain everything in simple words and keep it summarised till 2-3 lines.

Original Question: {original_question}
Step-back Question: {stepback_question}

General Context (for step-back question):
{stepback_context}

Specific Context (for original question):
{specific_context}

Instructions:
1. First, use the general context to understand the broader policy or concept
2. Then, use the specific context to answer the original question precisely
3. Provide a comprehensive answer that addresses the student's specific need
4. If information is not available in the context, clearly state that
5. Be helpful and student-friendly in your response


Answer:"""
        )
        
        self.synthesis_chain = LLMChain(
            llm=self.llm,
            prompt=self.synthesis_template
        )
    
    def generate_stepback_question(self, question: str) -> str:
        """Generate step-back question for broader context."""
        try:
            stepback_q = self.stepback_chain.run(question=question)
            return stepback_q.strip()
        except Exception as e:
            print(f"Step-back generation failed: {e}")
            return f"What are the general policies related to {question}?"
    
    def synthesize_answer(self, original_question: str, stepback_question: str, 
                         stepback_context: str, specific_context: str) -> str:
        """Synthesize final answer using both contexts."""
        try:
            answer = self.synthesis_chain.run(
                original_question=original_question,
                stepback_question=stepback_question,
                stepback_context=stepback_context,
                specific_context=specific_context
            )
            return answer.strip()
        except Exception as e:
            print(f"Answer synthesis failed: {e}")
            return "I apologize, but I encountered an error while processing your question. Please try again."

class CollegeHandbookRAG:
    """Complete RAG system for college handbook Q&A."""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.embedding_model = AdvancedEmbeddingModel(config.EMBEDDING_MODEL)
        self.search_engine = HybridSearchEngine(self.embedding_model)
        self.query_enhancer = QueryEnhancer(config.LLM_MODEL)
        self.stepback_prompter = StepBackPrompter(config.LLM_MODEL)
        self.is_indexed = False
    
    def load_and_index_handbook(self, file_path: str):
        """Load and index the college handbook."""
        print(f"Loading handbook from: {file_path}")
        
        # Load document
        text = self.doc_processor.load_document(file_path)
        print(f"Loaded document with {len(text)} characters")
        
        # Chunk document
        chunks = self.doc_processor.chunk_document(text, file_path)
        print(f"Created {len(chunks)} chunks")
        
        # Index for search
        self.search_engine.index_documents(chunks)
        self.is_indexed = True
        
        print("Handbook indexed successfully!")
        return len(chunks)
    
    def answer_question(self, question: str, use_query_enhancement: bool = True, 
                       use_stepback: bool = True, top_k: int = 5) -> Dict[str, Any]:
        """Answer student question using the complete RAG pipeline."""
        if not self.is_indexed:
            return {"error": "Please load and index the handbook first!"}
        
        print(f"\nProcessing question: {question}")
        
        # Step 1: Query Enhancement
        enhanced_query = question
        if use_query_enhancement:
            print("Enhancing query...")
            enhancement = self.query_enhancer.enhance_query(question)
            enhanced_query = self.query_enhancer.create_expanded_query(enhancement)
            print(f"Enhanced query: {enhanced_query}")
        
        # Step 2: Step-back prompting
        stepback_question = None
        stepback_context = ""
        
        if use_stepback:
            print("Generating step-back question...")
            stepback_question = self.stepback_prompter.generate_stepback_question(question)
            print(f"Step-back question: {stepback_question}")
            
            # Search for step-back context
            stepback_chunks = self.search_engine.hybrid_search(stepback_question, top_k=3)
            stepback_context = "\n\n".join([chunk.content for chunk in stepback_chunks])
        
        # Step 3: Hybrid search for specific question
        print("Performing hybrid search...")
        relevant_chunks = self.search_engine.hybrid_search(enhanced_query, top_k=top_k)
        specific_context = "\n\n".join([chunk.content for chunk in relevant_chunks])
        
        # Step 4: Generate final answer
        if use_stepback:
            answer = self.stepback_prompter.synthesize_answer(
                question, stepback_question, stepback_context, specific_context
            )
        else:
            # Use a simpler prompt for direct answers
            answer = self.stepback_prompter.synthesize_answer(
                question, question, specific_context, specific_context
            )
        
        return {
            "question": question,
            "enhanced_query": enhanced_query,
            "stepback_question": stepback_question,
            "answer": answer,
            "context": {
                "stepback": stepback_context,
                "specific": specific_context
            }
        }

# Example usage:
if __name__ == "__main__":
    try:
        # Initialize RAG system
        rag_system = CollegeHandbookRAG()
        logger.info("RAG system initialized successfully!")

        # Load and index the handbook
        handbook_path = config.DEFAULT_HANDBOOK_PATH
        logger.info(f"Loading handbook from: {handbook_path}")
        
        rag_system.load_and_index_handbook(handbook_path)
        
        # Interactive Q&A loop
        while True:
            try:
                question = input("\nEnter your question (or 'quit' to exit): ").strip()
                if not question:
                    continue
                    
                if question.lower() == 'quit':
                    logger.info("Exiting program...")
                    break
                
                response = rag_system.answer_question(question)
                print("\nAnswer:", response["answer"])
                
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                print(f"Error: {str(e)}")
                
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        print(f"Error: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        print(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Error: An unexpected error occurred: {str(e)}") 