"""
Trellix Knowledge RAG System
=============================

A comprehensive framework for building a RAG (Retrieval-Augmented Generation) system 
for Trellix's product offerings using Confluence PRDs, website data, and other documentation sources.

This framework:
1. Loads data from multiple sources (Confluence, websites, PRDs)
2. Segments textual content using RecursiveCharacterTextSplitter
3. Creates vector embeddings using Vertex AI Embeddings APIs
4. Stores documents and embeddings in ChromaDB vector database
5. Provides semantic search capabilities for RAG implementation
6. [OPTIONAL] Graph creation is commented out to simplify to vector-only RAG

Note: Graph building functionality is commented out to reduce complexity and focus on 
simple RAG implementation using vector similarity search.

Author: Senior Software Engineer
Date: September 2025
"""

import os
import json
import logging
import asyncio
import warnings
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import uuid

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

# Core dependencies
import pandas as pd
import numpy as np
from pathlib import Path

# Google Cloud dependencies - with fallbacks
try:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel
    from vertexai.language_models import TextEmbeddingModel
    from google.cloud import spanner
    VERTEX_AI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vertex AI not available: {e}")
    VERTEX_AI_AVAILABLE = False

try:
    from google.cloud import documentai
    DOCUMENT_AI_AVAILABLE = True
except ImportError:
    DOCUMENT_AI_AVAILABLE = False

# LangChain dependencies - with fallbacks
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_BASIC_AVAILABLE = True
except ImportError:
    print("Warning: Basic LangChain not available")
    LANGCHAIN_BASIC_AVAILABLE = False

try:
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_google_vertexai import VertexAI
    LANGCHAIN_ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced LangChain not available: {e}")
    LANGCHAIN_ADVANCED_AVAILABLE = False

# Web scraping and API dependencies - with fallbacks
try:
    from atlassian import Confluence
    CONFLUENCE_AVAILABLE = True
except ImportError:
    print("Warning: Confluence API not available")
    CONFLUENCE_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    print("Warning: Web scraping not available")
    WEB_SCRAPING_AVAILABLE = False

try:
    import scrapy
    from scrapy.crawler import CrawlerProcess
    SCRAPY_AVAILABLE = True
except ImportError:
    SCRAPY_AVAILABLE = False

# Graph and visualization dependencies - with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    print("Warning: NetworkX not available")
    NETWORKX_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    print("Warning: spaCy not available")
    SPACY_AVAILABLE = False

# Vector Database dependencies - Using Spanner instead of ChromaDB
# Spanner now supports vector embeddings natively
try:
    from google.cloud import spanner
    SPANNER_VECTOR_AVAILABLE = True
except ImportError:
    print("Warning: Spanner not available")
    SPANNER_VECTOR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeGraphConfig:
    """Configuration class for the Knowledge Graph Builder"""
    
    # Google Cloud Configuration
    project_id: str = "svc-hackathon-prod15"
    location: str = "us-central1"
    credentials_path: str = "svc-hackathon-prod15-534bc641841c.json"
    
    # Vertex AI Configuration - Optimized for Confluence Help Chatbot
    model_name: str = "gemini-2.0-flash-lite-001"
    embedding_model: str = "text-embedding-004"  # Best for technical documentation
    
    # Spanner Configuration
    spanner_instance_id: str = "trellix-knowledge-graph"
    spanner_database_id: str = "knowledge_graph_db"
    
    # Text Processing Configuration - Optimized for Help Content
    chunk_size: int = 512    # Optimal for technical help articles
    chunk_overlap: int = 64  # Good context preservation for Q&A
    max_tokens_per_chunk: int = 4000
    
    # Confluence Configuration - For Help Documentation
    confluence_url: str = "https://confluence.trellix.com"
    confluence_username: str = "aayush.choudhry@trellix.com"
    confluence_api_token: str = ""
    confluence_spaces: List[str] = field(default_factory=lambda: [
        "HELP",      # Help documentation
        "KB",        # Knowledge base
        "FAQ",       # Frequently asked questions
        "SUPPORT",   # Support articles
        "DOCS"       # Product documentation
    ])
    confluence_use_bearer_token: bool = True
    
    # Web Scraping Configuration
    trellix_domains: List[str] = field(default_factory=lambda: [
        "https://www.trellix.com",
        "https://docs.trellix.com",
        "https://community.trellix.com"
    ])
    
    # Graph Configuration
    max_graph_size: int = 10000
    similarity_threshold: float = 0.8
    
    # Database Configuration
    batch_size: int = 25  # Smaller batch size to minimize duplicate key conflicts
    max_retries: int = 3   # Maximum retries for failed batches
    
    # Vector Database Configuration - Using Spanner
    vector_collection_name: str = "trellix_knowledge_vectors"
    use_vector_db: bool = True  # Enable vector database for RAG
    vector_table_name: str = "trellix_document_embeddings"  # Existing Spanner table with data
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.project_id:
            raise ValueError("project_id is required")
        if not self.credentials_path or not os.path.exists(self.credentials_path):
            logger.warning(f"Credentials file not found: {self.credentials_path}")

class DataSource:
    """Base class for data sources"""
    
    def __init__(self, source_type: str, config: KnowledgeGraphConfig):
        self.source_type = source_type
        self.config = config
        self.documents: List[Document] = []
    
    async def extract_data(self) -> List[Document]:
        """Extract data from the source"""
        raise NotImplementedError("Subclasses must implement extract_data")
    
    def _create_document(self, content: str, metadata: Dict[str, Any]) -> Document:
        """Create a LangChain Document with proper metadata"""
        metadata.update({
            'source_type': self.source_type,
            'extraction_timestamp': datetime.now().isoformat(),
            'content_hash': hashlib.md5(content.encode()).hexdigest()
        })
        return Document(page_content=content, metadata=metadata)

class ConfluenceDataSource(DataSource):
    """Data source for Confluence pages and spaces"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        super().__init__("confluence", config)
        self.confluence = None
        self._initialize_confluence()
    
    def _initialize_confluence(self):
        """Initialize Confluence API client with Bearer token support"""
        try:
            if not CONFLUENCE_AVAILABLE:
                logger.warning("Confluence API not available")
                return
                
            if all([self.config.confluence_url, 
                   self.config.confluence_username, 
                   self.config.confluence_api_token]):
                
                if self.config.confluence_use_bearer_token:
                    # Use Bearer token authentication (for modern Confluence)
                    self.confluence = Confluence(
                        url=self.config.confluence_url,
                        token=self.config.confluence_api_token
                    )
                    logger.info("Confluence client initialized with Bearer token")
                else:
                    # Use Basic authentication (traditional method)
                    self.confluence = Confluence(
                        url=self.config.confluence_url,
                        username=self.config.confluence_username,
                        password=self.config.confluence_api_token
                    )
                    logger.info("Confluence client initialized with Basic auth")
            else:
                logger.warning("Confluence credentials not provided")
        except Exception as e:
            logger.error(f"Failed to initialize Confluence client: {e}")
            logger.info("Try switching between Bearer token and Basic auth methods")
    
    async def extract_data(self) -> List[Document]:
        """Extract data from Confluence spaces"""
        if not self.confluence:
            logger.warning("Confluence client not available")
            return []
        
        documents = []
        
        try:
            for space_key in self.config.confluence_spaces:
                logger.info(f"Extracting data from Confluence space: {space_key}")
                
                # Get all pages in the space
                pages = self.confluence.get_all_pages_from_space(
                    space=space_key,
                    start=0,
                    limit=1000,
                    expand='body.storage,version,metadata'
                )
                
                for page in pages:
                    content = self._extract_page_content(page)
                    if content.strip():
                        metadata = {
                            'page_id': page['id'],
                            'title': page['title'],
                            'space': space_key,
                            'url': f"{self.config.confluence_url}/pages/{page['id']}",
                            'version': page.get('version', {}).get('number', 1),
                            'last_modified': page.get('version', {}).get('when')
                        }
                        documents.append(self._create_document(content, metadata))
                
                logger.info(f"Extracted {len(documents)} pages from space {space_key}")
        
        except Exception as e:
            logger.error(f"Error extracting Confluence data: {e}")
        
        self.documents = documents
        return documents
    
    def _extract_page_content(self, page: Dict[str, Any]) -> str:
        """Extract clean text content from a Confluence page"""
        try:
            # Get the storage format content
            body = page.get('body', {}).get('storage', {}).get('value', '')
            
            # Parse HTML and extract text
            soup = BeautifulSoup(body, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'meta', 'link']):
                element.decompose()
            
            # Extract text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            text = ' '.join(text.split())
            
            return text
        
        except Exception as e:
            logger.error(f"Error extracting content from page {page.get('id')}: {e}")
            return ""

class WebsiteDataSource(DataSource):
    """Data source for website content scraping"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        super().__init__("website", config)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Trellix Knowledge Graph Builder 1.0'
        })
    
    async def extract_data(self) -> List[Document]:
        """Extract data from Trellix websites"""
        documents = []
        
        for domain in self.config.trellix_domains:
            logger.info(f"Scraping domain: {domain}")
            try:
                domain_docs = await self._scrape_domain(domain)
                documents.extend(domain_docs)
            except Exception as e:
                logger.error(f"Error scraping domain {domain}: {e}")
        
        self.documents = documents
        return documents
    
    async def _scrape_domain(self, domain: str) -> List[Document]:
        """Scrape a specific domain for content"""
        documents = []
        visited_urls = set()
        urls_to_visit = [domain]
        max_pages = 100  # Limit to prevent excessive scraping
        
        while urls_to_visit and len(visited_urls) < max_pages:
            url = urls_to_visit.pop(0)
            
            if url in visited_urls:
                continue
            
            visited_urls.add(url)
            
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    content = self._extract_web_content(response.text, url)
                    if content.strip():
                        metadata = {
                            'url': url,
                            'domain': domain,
                            'status_code': response.status_code,
                            'content_type': response.headers.get('content-type', '')
                        }
                        documents.append(self._create_document(content, metadata))
                    
                    # Extract additional URLs to visit
                    new_urls = self._extract_links(response.text, domain)
                    urls_to_visit.extend(new_urls[:10])  # Limit new URLs
                
            except Exception as e:
                logger.error(f"Error scraping URL {url}: {e}")
        
        logger.info(f"Scraped {len(documents)} pages from domain {domain}")
        return documents
    
    def _extract_web_content(self, html: str, url: str) -> str:
        """Extract clean text content from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Focus on main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            text = ' '.join(text.split())
            
            return text
        
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return ""
    
    def _extract_links(self, html: str, base_domain: str) -> List[str]:
        """Extract relevant links from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith(base_domain) or href.startswith('/'):
                    if href.startswith('/'):
                        href = base_domain + href
                    links.append(href)
            
            return list(set(links))  # Remove duplicates
        
        except Exception as e:
            logger.error(f"Error extracting links: {e}")
            return []

class DocumentProcessor:
    """Process and segment documents for knowledge graph creation"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.text_splitter = None
        self.nlp = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize text processing components with fallbacks"""
        # Initialize text splitter
        if LANGCHAIN_BASIC_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
            )
        else:
            logger.warning("LangChain not available, using basic text splitting")
        
        # Load spaCy model
        self._load_nlp_model()
    
    def _load_nlp_model(self):
        """Load spaCy NLP model for text processing"""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available")
            return
            
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
    
    async def process_documents(self, documents: List) -> List:
        """Process and segment documents into chunks"""
        if not documents:
            return []
        
        processed_docs = []
        
        # Handle case where Document class is not available
        if not LANGCHAIN_BASIC_AVAILABLE:
            logger.warning("LangChain Document class not available, using basic processing")
            return self._basic_process_documents(documents)
        
        for doc in documents:
            try:
                # Clean and preprocess text
                cleaned_content = self._clean_text(doc.page_content)
                
                # Split into chunks
                if self.text_splitter:
                    chunks = self.text_splitter.split_text(cleaned_content)
                else:
                    chunks = self._basic_split_text(cleaned_content)
                
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 50:  # Filter out very short chunks
                        chunk_metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                        chunk_metadata.update({
                            'chunk_id': f"{chunk_metadata.get('content_hash', uuid.uuid4().hex)}_{i}",
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'chunk_size': len(chunk)
                        })
                        
                        # Create document object
                        if LANGCHAIN_BASIC_AVAILABLE:
                            processed_docs.append(Document(
                                page_content=chunk,
                                metadata=chunk_metadata
                            ))
                        else:
                            processed_docs.append({
                                'content': chunk,
                                'metadata': chunk_metadata
                            })
            
            except Exception as e:
                logger.error(f"Error processing document: {e}")
        
        logger.info(f"Processed {len(documents)} documents into {len(processed_docs)} chunks")
        return processed_docs
    
    def _basic_process_documents(self, documents: List) -> List:
        """Basic document processing without LangChain"""
        processed_docs = []
        
        for doc in documents:
            try:
                content = doc if isinstance(doc, str) else str(doc)
                cleaned_content = self._clean_text(content)
                chunks = self._basic_split_text(cleaned_content)
                
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 50:
                        processed_docs.append({
                            'content': chunk,
                            'metadata': {
                                'chunk_id': f"{uuid.uuid4().hex}_{i}",
                                'chunk_index': i,
                                'total_chunks': len(chunks),
                                'chunk_size': len(chunk)
                            }
                        })
                        
            except Exception as e:
                logger.error(f"Error in basic processing: {e}")
        
        return processed_docs
    
    def _basic_split_text(self, text: str) -> List[str]:
        """Basic text splitting without LangChain"""
        # Simple sentence-based splitting
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove special characters that might interfere with processing
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        text = text.replace('\u200b', '')   # Zero-width space
        
        return text.strip()

class VertexAIGraphBuilder:
    """Build knowledge graphs using Vertex AI and LangChain"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.llm = None
        self.embedding_model = None
        self.graph_transformer = None
        self.initialized = False
        self.fallback_mode = False
    
    async def initialize(self):
        """Initialize Vertex AI services with fallbacks"""
        try:
            if not VERTEX_AI_AVAILABLE:
                logger.warning("Vertex AI not available, switching to fallback mode")
                self.fallback_mode = True
                self.initialized = True
                return
            
            # Set credentials
            if os.path.exists(self.config.credentials_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.credentials_path
            else:
                logger.warning(f"Credentials file not found: {self.config.credentials_path}")
            
            # Suppress deprecation warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Initialize Vertex AI
                vertexai.init(project=self.config.project_id, location=self.config.location)
                
                # Initialize LLM
                if LANGCHAIN_ADVANCED_AVAILABLE:
                    self.llm = VertexAI(
                        model_name=self.config.model_name,
                        project=self.config.project_id,
                        location=self.config.location
                    )
                
                # Initialize embedding model
                try:
                    self.embedding_model = TextEmbeddingModel.from_pretrained(self.config.embedding_model)
                    logger.info(f"Embedding model {self.config.embedding_model} initialized successfully")
                except Exception as emb_error:
                    logger.warning(f"Failed to initialize embedding model {self.config.embedding_model}: {emb_error}")
                    # Try fallback model
                    try:
                        fallback_model = "textembedding-gecko@latest"
                        self.embedding_model = TextEmbeddingModel.from_pretrained(fallback_model)
                        logger.info(f"Fallback embedding model {fallback_model} initialized successfully")
                    except Exception as fallback_error:
                        logger.error(f"Fallback embedding model also failed: {fallback_error}")
                        self.fallback_mode = True
                
                # Initialize graph transformer
                if LANGCHAIN_ADVANCED_AVAILABLE and self.llm:
                    try:
                        self.graph_transformer = LLMGraphTransformer(
                            llm=self.llm,
                            allowed_nodes=["Product", "Feature", "Component", "Process", "Person", "Organization"],
                            allowed_relationships=["USES", "IMPLEMENTS", "RELATES_TO", "PART_OF", "MANAGES", "DEPENDS_ON"]
                        )
                        logger.info("LLMGraphTransformer initialized successfully")
                        
                        # Test available methods
                        transformer_methods = [method for method in dir(self.graph_transformer) if not method.startswith('_')]
                        logger.debug(f"Available transformer methods: {transformer_methods}")
                        
                    except ImportError as e:
                        logger.error(f"Failed to initialize LLMGraphTransformer: {e}")
                        logger.info("Please install: pip install json-repair")
                        self.fallback_mode = True
                    except Exception as e:
                        logger.error(f"Unexpected error initializing LLMGraphTransformer: {e}")
                        self.fallback_mode = True
                else:
                    logger.warning("LLMGraphTransformer not available - missing dependencies")
                    self.fallback_mode = True
            
            self.initialized = True
            
            if self.fallback_mode:
                logger.warning("Vertex AI Graph Builder initialized in fallback mode")
            else:
                logger.info("Vertex AI Graph Builder initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI Graph Builder: {e}")
            self.fallback_mode = True
            self.initialized = True
    
    async def build_graph(self, documents: List) -> Any:
        """Build knowledge graph from processed documents"""
        if not self.initialized:
            await self.initialize()
        
        if not NETWORKX_AVAILABLE:
            logger.error("NetworkX not available, cannot build graph")
            return self._create_fallback_graph(documents)
        
        graph = nx.MultiDiGraph()
        
        if self.fallback_mode or not self.graph_transformer:
            logger.info("Using fallback graph construction method")
            return self._build_fallback_graph(documents, graph)
        
        # Process documents in batches to avoid API limits
        batch_size = 10
        successful_batches = 0
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i//batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            try:
                # # Try AI-powered graph transformation first
                # if not self.fallback_mode and self.graph_transformer:
                #     success = self._try_ai_graph_processing(batch, graph)
                #     if success:
                #         successful_batches += 1
                #         continue
                
                # Fallback to simple processing
                self._add_fallback_nodes_for_batch(batch, graph, i)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Add fallback nodes for this batch
                self._add_fallback_nodes_for_batch(batch, graph, i)
                continue
        
        logger.info(f"Built knowledge graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        logger.info(f"Successfully processed {successful_batches}/{total_batches} batches with AI")
        return graph
    
    def _try_ai_graph_processing(self, batch, graph) -> bool:
        """Try to process batch with AI graph transformer, return success status"""
        try:
            # Transform documents to graph elements
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Ensure all documents in batch are Document objects
                doc_batch = []
                for doc in batch:
                    if LANGCHAIN_BASIC_AVAILABLE:
                        if hasattr(doc, 'page_content'):
                            doc_batch.append(doc)
                        elif isinstance(doc, dict):
                            # Convert dict to Document object
                            content = doc.get('content', str(doc))
                            metadata = doc.get('metadata', {})
                            doc_batch.append(Document(page_content=content, metadata=metadata))
                        else:
                            # Convert string or other to Document object
                            content = str(doc)
                            doc_batch.append(Document(page_content=content, metadata={}))
                    else:
                        # No LangChain, can't use AI processing
                        return False
                
                if not doc_batch:
                    return False
                
                # Debug: Check what we're passing to the transformer
                logger.debug(f"Passing {len(doc_batch)} documents to graph transformer")
                
                # Try different method names depending on LangChain version
                graph_documents = []
                if hasattr(self.graph_transformer, 'convert_to_graph_documents'):
                    # Newer version
                    logger.debug("Using convert_to_graph_documents method")
                    graph_documents = self.graph_transformer.convert_to_graph_documents(doc_batch)
                elif hasattr(self.graph_transformer, 'transform_documents'):
                    # Older version
                    logger.debug("Using transform_documents method") 
                    graph_documents = self.graph_transformer.transform_documents(doc_batch)
                else:
                    logger.debug("No known transformer method found")
                    return False
            
            # Add nodes and relationships to graph
            for graph_doc in graph_documents:
                self._add_graph_elements_to_networkx(graph, graph_doc)
            
            return True
            
        except Exception as e:
            logger.warning(f"AI graph processing failed: {e}. Falling back to simple processing.")
            return False
    
    def _add_fallback_nodes_for_batch(self, batch, graph, batch_start_index):
        """Add simple fallback nodes for a batch of documents"""
        for j, doc in enumerate(batch):
            try:
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                elif isinstance(doc, dict) and 'content' in doc:
                    content = doc['content']
                elif isinstance(doc, dict):
                    content = str(doc)
                else:
                    content = str(doc)
                
                doc_id = f"doc_{batch_start_index + j}_{uuid.uuid4().hex[:8]}"
                graph.add_node(doc_id, type="Document", content=content[:200])
                
                # Add some simple keyword-based relationships
                # todo: enhance more
                keywords = ["integrations", "cases","alert","product", "feature", "security", "endpoint", "network", "api", "siem", "CHI", "helix", "integration"]
                for keyword in keywords:
                    if keyword.lower() in content.lower():
                        keyword_id = f"keyword_{keyword}"
                        if not graph.has_node(keyword_id):
                            graph.add_node(keyword_id, type="Keyword", name=keyword.title())
                        graph.add_edge(doc_id, keyword_id, type="MENTIONS")
                
            except Exception as sub_e:
                logger.error(f"Error adding fallback node: {sub_e}")
    
    def _build_fallback_graph(self, documents: List, graph) -> Any:
        """Build a simple graph without AI processing"""
        logger.info("Building fallback knowledge graph...")
        
        # Simple keyword-based graph construction
        keywords = ["product", "feature", "security", "endpoint", "network", "api", "siem"]
        
        for i, doc in enumerate(documents):
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            elif isinstance(doc, dict) and 'content' in doc:
                content = doc['content']
            elif isinstance(doc, dict):
                content = str(doc)
            else:
                content = str(doc)
            
            # Add document node
            doc_id = f"doc_{i}"
            graph.add_node(doc_id, type="Document", name=f"Document {i}", content=content[:100])
            
            # Add keyword nodes and relationships
            for keyword in keywords:
                if keyword.lower() in content.lower():
                    keyword_id = f"keyword_{keyword}"
                    if not graph.has_node(keyword_id):
                        graph.add_node(keyword_id, type="Keyword", name=keyword.title())
                    graph.add_edge(doc_id, keyword_id, type="MENTIONS")
        
        logger.info(f"Built fallback graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph
    
    def _create_fallback_graph(self, documents: List) -> Dict:
        """Create a simple dictionary-based graph when NetworkX is not available"""
        return {
            'nodes': [{'id': f'doc_{i}', 'type': 'Document'} for i in range(len(documents))],
            'edges': [],
            'metadata': {'fallback': True, 'total_documents': len(documents)}
        }
    
    def _add_graph_elements_to_networkx(self, graph, graph_document):
        """Add graph elements from LangChain graph document to NetworkX graph"""
        try:
            # Handle nodes
            if hasattr(graph_document, 'nodes'):
                for node in graph_document.nodes:
                    node_id = getattr(node, 'id', str(node))
                    node_type = getattr(node, 'type', 'Entity')
                    node_properties = getattr(node, 'properties', {})
                    
                    graph.add_node(
                        node_id,
                        type=node_type,
                        name=node_id,
                        **node_properties
                    )
            
            # Handle relationships
            if hasattr(graph_document, 'relationships'):
                for rel in graph_document.relationships:
                    source = getattr(rel, 'source', None)
                    target = getattr(rel, 'target', None)
                    rel_type = getattr(rel, 'type', 'RELATED_TO')
                    rel_properties = getattr(rel, 'properties', {})
                    
                    if source and target:
                        # Get actual node IDs
                        source_id = getattr(source, 'id', str(source))
                        target_id = getattr(target, 'id', str(target))
                        
                        graph.add_edge(
                            source_id,
                            target_id,
                            type=rel_type,
                            **rel_properties
                        )
                        
        except Exception as e:
            logger.error(f"Error adding graph elements to NetworkX: {e}")
            # Fallback: create simple nodes from document content
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            graph.add_node(doc_id, type="Document", name="Processed Document")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        if not self.initialized:
            await self.initialize()
        
        if self.fallback_mode or not self.embedding_model:
            logger.warning("Using fallback embeddings (random vectors)")
            return [[0.1] * 768 for _ in texts]  # Dummy embeddings
        
        embeddings = []
        batch_size = 100  # Vertex AI embedding API batch limit
        max_tokens_per_text = 18000  # Conservative limit (model supports 20k)
        
        # Truncate texts that are too long
        processed_texts = []
        for text in texts:
            if len(text.split()) > max_tokens_per_text // 4:  # Rough token estimation
                # Truncate to approximate token limit
                words = text.split()
                truncated_text = ' '.join(words[:max_tokens_per_text // 4])
                processed_texts.append(truncated_text)
                logger.debug(f"Truncated text from {len(words)} to {len(truncated_text.split())} words")
            else:
                processed_texts.append(text)
        
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i + batch_size]
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    batch_embeddings = self.embedding_model.get_embeddings(batch)
                embeddings.extend([emb.values for emb in batch_embeddings])
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Add zero embeddings as fallback
                embeddings.extend([[0.0] * 768] * len(batch))
        
        logger.info(f"Generated embeddings for {len(processed_texts)} text chunks")
        return embeddings

class SpannerVectorDatabase:
    """Spanner-based vector database for storing document embeddings for RAG"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.spanner_client = None
        self.instance = None
        self.database = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize Spanner client for vector operations"""
        try:
            if not SPANNER_VECTOR_AVAILABLE:
                logger.warning("Spanner not available, skipping vector database initialization")
                return False
            
            # Initialize Spanner client
            self.spanner_client = spanner.Client()
            self.instance = self.spanner_client.instance(self.config.spanner_instance_id)
            self.database = self.instance.database(self.config.spanner_database_id)
            
            # Create vector embeddings table if it doesn't exist
            await self._ensure_vector_table_exists()
            
            self.initialized = True
            logger.info("✅ Spanner Vector Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Spanner vector database: {e}")
            return False
    
    async def _ensure_vector_table_exists(self):
        """Ensure the vector embeddings table exists in Spanner"""
        try:
            # Check if table exists
            with self.database.snapshot() as snapshot:
                try:
                    results = snapshot.execute_sql(
                        f"SELECT 1 FROM {self.config.vector_table_name} LIMIT 1"
                    )
                    list(results)  # Try to consume results
                    logger.info(f"Vector table {self.config.vector_table_name} already exists")
                    return
                except Exception:
                    # Table doesn't exist, create it
                    pass
            
            # Create the vector embeddings table with correct Spanner types
            ddl = f"""
            CREATE TABLE {self.config.vector_table_name} (
                doc_id STRING(MAX) NOT NULL,
                content STRING(MAX),
                embedding ARRAY<FLOAT64>,
                metadata JSON,
                created_at TIMESTAMP OPTIONS (allow_commit_timestamp=true),
                source_type STRING(100),
                title STRING(500)
            ) PRIMARY KEY (doc_id)
            """
            
            operation = self.database.update_ddl([ddl])
            operation.result()  # Wait for completion
            logger.info(f"Created vector table: {self.config.vector_table_name}")
            
        except Exception as e:
            logger.error(f"Error ensuring vector table exists: {e}")
            raise
    
    async def store_documents(self, documents: List, embeddings: List[List[float]]) -> bool:
        """Store documents and their embeddings in Spanner"""
        try:
            if not self.initialized or not self.database:
                logger.error("Spanner vector database not initialized")
                return False
            
            # Prepare data for batch insert
            rows_to_insert = []
            
            for i, doc in enumerate(documents):
                if i >= len(embeddings):
                    break
                    
                doc_id = f"doc_{hashlib.md5(str(doc).encode()).hexdigest()[:16]}"
                
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    metadata = dict(doc.metadata) if hasattr(doc, 'metadata') else {}
                elif isinstance(doc, dict):
                    content = doc.get('content', str(doc))
                    metadata = doc.get('metadata', {})
                else:
                    content = str(doc)
                    metadata = {}
                
                # Add processing timestamp
                metadata.update({
                    'stored_at': datetime.now().isoformat(),
                    'doc_index': i
                })
                
                rows_to_insert.append((
                    doc_id,
                    content,
                    embeddings[i],
                    json.dumps(metadata),
                    spanner.COMMIT_TIMESTAMP,  # Fixed: Use COMMIT_TIMESTAMP
                    metadata.get('source_type', 'unknown'),
                    metadata.get('title', 'Untitled')
                ))
            
            # Batch insert into Spanner
            with self.database.batch() as batch:
                batch.insert(
                    table=self.config.vector_table_name,
                    columns=[
                        'doc_id', 'content', 'embedding', 'metadata', 
                        'created_at', 'source_type', 'title'
                    ],
                    values=rows_to_insert
                )
            
            logger.info(f"✅ Stored {len(rows_to_insert)} documents in Spanner vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store documents in Spanner vector database: {e}")
            return False
    
    def search_similar(self, query: str, n_results: int = 5, query_embedding: List[float] = None) -> List[Dict]:
        """Search for similar documents using Spanner vector similarity"""
        try:
            if not self.initialized or not self.database or not query_embedding:
                logger.error("Spanner vector database not initialized or no query embedding provided")
                return []
            
            # Use Spanner's vector similarity search (approximate)
            # Note: This is a simplified approach - in production, you might want to use 
            # Spanner's vector search capabilities or implement cosine similarity
            
            with self.database.snapshot() as snapshot:
                # For now, we'll get all documents and compute similarity in memory
                # In production, you'd use Spanner's built-in vector similarity functions
                results = snapshot.execute_sql(
                    f"""
                    SELECT doc_id, content, embedding, metadata, title, source_type
                    FROM {self.config.vector_table_name}
                    LIMIT 100
                    """
                )
                
                similar_docs = []
                for row in results:
                    doc_embedding = row[2]  # embedding column
                    
                    # Compute cosine similarity
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    
                    # Handle Spanner JsonObject properly
                    metadata_value = row[3]
                    if metadata_value:
                        # Spanner JSON objects need special handling
                        try:
                            if hasattr(metadata_value, '__dict__'):
                                # It's a JsonObject, convert to dict
                                metadata = dict(metadata_value)
                            elif isinstance(metadata_value, str):
                                # It's a JSON string, parse it
                                metadata = json.loads(metadata_value)
                            else:
                                # It's already a dict or dict-like
                                metadata = dict(metadata_value)
                        except (TypeError, ValueError, AttributeError) as e:
                            logger.warning(f"Could not parse metadata: {e}")
                            metadata = {}
                    else:
                        metadata = {}
                    
                    similar_docs.append({
                        'id': row[0],
                        'text': row[1],
                        'embedding': doc_embedding,
                        'metadata': metadata,
                        'title': row[4],
                        'source_type': row[5],
                        'similarity': similarity,
                        'distance': 1 - similarity
                    })
                
                # Sort by similarity and return top results
                similar_docs.sort(key=lambda x: x['similarity'], reverse=True)
                return similar_docs[:n_results]
                
        except Exception as e:
            logger.error(f"Failed to search Spanner vector database: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        try:
            import numpy as np
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0.0
            
            return dot_product / (norm_vec1 * norm_vec2)
        except Exception:
            return 0.0
    
    def get_collection_info(self) -> Dict:
        """Get information about the vector database collection"""
        try:
            if not self.initialized or not self.database:
                return {"status": "not_initialized"}
            
            with self.database.snapshot() as snapshot:
                results = snapshot.execute_sql(
                    f"SELECT COUNT(*) FROM {self.config.vector_table_name}"
                )
                count = list(results)[0][0]
            
            return {
                "status": "initialized",
                "table_name": self.config.vector_table_name,
                "document_count": count,
                "database": self.config.spanner_database_id,
                "instance": self.config.spanner_instance_id
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"status": "error", "error": str(e)}

class SpannerGraphStore:
    """Store and manage knowledge graph in Google Cloud Spanner"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.spanner_client = None
        self.instance = None
        self.database = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize Spanner connection and create database if needed"""
        try:
            # Set credentials
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.credentials_path
            
            # Initialize Spanner client
            self.spanner_client = spanner.Client(project=self.config.project_id)
            self.instance = self.spanner_client.instance(self.config.spanner_instance_id)
            
            # Create database if it doesn't exist
            await self._ensure_database_exists()
            
            self.database = self.instance.database(self.config.spanner_database_id)
            self.initialized = True
            
            logger.info("Spanner Graph Store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Spanner Graph Store: {e}")
            raise
    
    async def _ensure_database_exists(self):
        """Ensure the Spanner database and tables exist"""
        try:
            # Check if database exists
            database = self.instance.database(self.config.spanner_database_id)
            
            if not database.exists():
                logger.info("Creating Spanner database...")
                
                # Create database first (without schema)
                operation = database.create()
                operation.result(timeout=300)  # Wait up to 5 minutes
                
                logger.info("Spanner database created successfully")
                
                # Now create the schema using DDL operations
                await self._create_database_schema(database)
            else:
                logger.info("Spanner database already exists")
                # Check if tables exist, create them if they don't
                await self._ensure_tables_exist(database)
                
        except Exception as e:
            logger.error(f"Error ensuring database exists: {e}")
            # Try to clean up if database was partially created
            try:
                database = self.instance.database(self.config.spanner_database_id)
                if database.exists():
                    logger.info("Attempting to drop and recreate database due to schema error...")
                    database.drop()
                    # Wait a bit for cleanup
                    import time
                    time.sleep(5)
                    # Try to create again with corrected schema
                    operation = database.create()
                    operation.result(timeout=300)
                    await self._create_database_schema(database)
                    logger.info("Database recreated successfully")
                else:
                    raise
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up and recreate database: {cleanup_error}")
                raise e
    
    async def _create_database_schema(self, database):
        """Create the database schema with all required tables"""
        try:
            logger.info("Creating database schema...")
            
            # Define DDL statements for creating tables
            ddl_statements = [
                """
                CREATE TABLE Nodes (
                    node_id STRING(36) NOT NULL,
                    node_type STRING(50) NOT NULL,
                    name STRING(500) NOT NULL,
                    properties JSON,
                    embedding ARRAY<FLOAT64>,
                    source_document_id STRING(100),
                    created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                    updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
                ) PRIMARY KEY (node_id)
                """,
                """
                CREATE TABLE Relationships (
                    relationship_id STRING(36) NOT NULL,
                    source_node_id STRING(36) NOT NULL,
                    target_node_id STRING(36) NOT NULL,
                    relationship_type STRING(50) NOT NULL,
                    properties JSON,
                    confidence_score FLOAT64,
                    created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                    FOREIGN KEY (source_node_id) REFERENCES Nodes (node_id),
                    FOREIGN KEY (target_node_id) REFERENCES Nodes (node_id)
                ) PRIMARY KEY (relationship_id)
                """,
                """
                CREATE TABLE Documents (
                    document_id STRING(36) NOT NULL,
                    source_type STRING(50) NOT NULL,
                    title STRING(500),
                    content STRING(MAX) NOT NULL,
                    metadata JSON,
                    content_hash STRING(32) NOT NULL,
                    created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                    updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
                ) PRIMARY KEY (document_id)
                """,
                """
                CREATE TABLE TextChunks (
                    chunk_id STRING(36) NOT NULL,
                    document_id STRING(36) NOT NULL,
                    chunk_index INT64 NOT NULL,
                    content STRING(MAX) NOT NULL,
                    embedding ARRAY<FLOAT64>,
                    metadata JSON,
                    created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                    FOREIGN KEY (document_id) REFERENCES Documents (document_id)
                ) PRIMARY KEY (chunk_id)
                """
            ]
            
            # Execute DDL statements to create tables
            operation = database.update_ddl(ddl_statements)
            operation.result(timeout=300)  # Wait up to 5 minutes
            
            logger.info("Database schema created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database schema: {e}")
            raise
    
    async def _create_missing_tables(self, database, missing_tables: set):
        """Create only the missing tables"""
        try:
            logger.info(f"Creating missing tables: {missing_tables}")
            
            # Define DDL statements for individual tables
            table_ddl = {
                'Documents': """
                    CREATE TABLE Documents (
                        document_id STRING(36) NOT NULL,
                        source_type STRING(50) NOT NULL,
                        title STRING(500),
                        content STRING(MAX) NOT NULL,
                        metadata JSON,
                        content_hash STRING(32) NOT NULL,
                        created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                        updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
                    ) PRIMARY KEY (document_id)
                """,
                'TextChunks': """
                    CREATE TABLE TextChunks (
                        chunk_id STRING(36) NOT NULL,
                        document_id STRING(36) NOT NULL,
                        chunk_index INT64 NOT NULL,
                        content STRING(MAX) NOT NULL,
                        embedding ARRAY<FLOAT64>,
                        metadata JSON,
                        created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                        FOREIGN KEY (document_id) REFERENCES Documents (document_id)
                    ) PRIMARY KEY (chunk_id)
                """,
                'Nodes': """
                    CREATE TABLE Nodes (
                        node_id STRING(36) NOT NULL,
                        node_type STRING(50) NOT NULL,
                        name STRING(500) NOT NULL,
                        properties JSON,
                        embedding ARRAY<FLOAT64>,
                        source_document_id STRING(100),
                        created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                        updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
                    ) PRIMARY KEY (node_id)
                """,
                'Relationships': """
                    CREATE TABLE Relationships (
                        relationship_id STRING(36) NOT NULL,
                        source_node_id STRING(36) NOT NULL,
                        target_node_id STRING(36) NOT NULL,
                        relationship_type STRING(50) NOT NULL,
                        properties JSON,
                        confidence_score FLOAT64,
                        created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                        FOREIGN KEY (source_node_id) REFERENCES Nodes (node_id),
                        FOREIGN KEY (target_node_id) REFERENCES Nodes (node_id)
                    ) PRIMARY KEY (relationship_id)
                """
            }
            
            # Create only missing tables
            ddl_statements = []
            for table_name in missing_tables:
                if table_name in table_ddl:
                    ddl_statements.append(table_ddl[table_name])
            
            if ddl_statements:
                operation = database.update_ddl(ddl_statements)
                operation.result(timeout=300)  # Wait up to 5 minutes
                logger.info(f"Successfully created missing tables: {missing_tables}")
            
        except Exception as e:
            logger.error(f"Error creating missing tables: {e}")
            raise
    
    async def _verify_table_schema(self, database):
        """Verify that all tables have the correct schema and add missing columns"""
        try:
            logger.info("Verifying table schema...")
            with database.snapshot() as snapshot:
                # Check Documents table columns
                try:
                    results = snapshot.execute_sql(
                        "SELECT column_name FROM information_schema.columns WHERE table_name = 'Documents'"
                    )
                    existing_columns = {row[0] for row in results}
                    logger.info(f"Existing columns in Documents table: {existing_columns}")
                    
                    required_columns = {'document_id', 'source_type', 'title', 'content', 'metadata', 'content_hash', 'created_at', 'updated_at'}
                    missing_columns = required_columns - existing_columns
                    
                    if missing_columns:
                        logger.info(f"Documents table is missing columns: {missing_columns}")
                        logger.info("Recreating Documents table with correct schema...")
                        
                        # Drop and recreate the Documents table with correct schema
                        ddl_statements = [
                            "DROP TABLE Documents",
                            """
                            CREATE TABLE Documents (
                                document_id STRING(36) NOT NULL,
                                source_type STRING(50) NOT NULL,
                                title STRING(500),
                                content STRING(MAX) NOT NULL,
                                metadata JSON,
                                content_hash STRING(32) NOT NULL,
                                created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                                updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
                            ) PRIMARY KEY (document_id)
                            """
                        ]
                        
                        operation = database.update_ddl(ddl_statements)
                        operation.result(timeout=300)
                        logger.info("Successfully recreated Documents table with correct schema")
                    else:
                        logger.info("Documents table schema is correct")
                            
                except Exception as e:
                    logger.warning(f"Could not verify Documents table schema: {e}")

                # Check Nodes table columns
                try:
                    results = snapshot.execute_sql(
                        "SELECT column_name FROM information_schema.columns WHERE table_name = 'Nodes'"
                    )
                    existing_columns = {row[0] for row in results}
                    logger.info(f"Existing columns in Nodes table: {existing_columns}")
                    
                    required_columns = {'node_id', 'node_type', 'name', 'properties', 'embedding', 'source_document_id', 'created_at', 'updated_at'}
                    missing_columns = required_columns - existing_columns
                    
                    if missing_columns:
                        logger.info(f"Nodes table is missing columns: {missing_columns}")
                        logger.info("Recreating Nodes table with correct schema...")
                        
                        # Drop and recreate the Nodes table with correct schema
                        ddl_statements = [
                            "DROP TABLE IF EXISTS Relationships",  # Drop relationships first due to foreign key
                            "DROP TABLE IF EXISTS Nodes",
                            """
                            CREATE TABLE Nodes (
                                node_id STRING(36) NOT NULL,
                                node_type STRING(50) NOT NULL,
                                name STRING(500) NOT NULL,
                                properties JSON,
                                embedding ARRAY<FLOAT64>,
                                source_document_id STRING(100),
                                created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                                updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
                            ) PRIMARY KEY (node_id)
                            """,
                            """
                            CREATE TABLE Relationships (
                                relationship_id STRING(36) NOT NULL,
                                source_node_id STRING(36) NOT NULL,
                                target_node_id STRING(36) NOT NULL,
                                relationship_type STRING(50) NOT NULL,
                                properties JSON,
                                confidence_score FLOAT64,
                                created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                                FOREIGN KEY (source_node_id) REFERENCES Nodes (node_id),
                                FOREIGN KEY (target_node_id) REFERENCES Nodes (node_id)
                            ) PRIMARY KEY (relationship_id)
                            """
                        ]
                        
                        operation = database.update_ddl(ddl_statements)
                        operation.result(timeout=300)
                        logger.info("Successfully recreated Nodes and Relationships tables with correct schema")
                    else:
                        logger.info("Nodes table schema is correct")
                        
                except Exception as e:
                    logger.warning(f"Could not verify Nodes table schema: {e}")
                    
        except Exception as e:
            logger.error(f"Error verifying table schema: {e}")
            # Don't raise here, continue with execution
    
    async def _force_schema_recreation(self, database):
        """Force recreation of all tables with correct schema"""
        try:
            logger.info("Forcing complete schema recreation...")
            
            # Drop all tables in the correct order (considering foreign keys)
            drop_statements = [
                "DROP TABLE IF EXISTS Relationships",
                "DROP TABLE IF EXISTS TextChunks", 
                "DROP TABLE IF EXISTS Nodes",
                "DROP TABLE IF EXISTS Documents"
            ]
            
            # Create all tables with correct schema
            create_statements = [
                """
                CREATE TABLE Documents (
                    document_id STRING(36) NOT NULL,
                    source_type STRING(50) NOT NULL,
                    title STRING(500),
                    content STRING(MAX) NOT NULL,
                    metadata JSON,
                    content_hash STRING(32) NOT NULL,
                    created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                    updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
                ) PRIMARY KEY (document_id)
                """,
                """
                CREATE TABLE Nodes (
                    node_id STRING(36) NOT NULL,
                    node_type STRING(50) NOT NULL,
                    name STRING(500) NOT NULL,
                    properties JSON,
                    embedding ARRAY<FLOAT64>,
                    source_document_id STRING(100),
                    created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                    updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
                ) PRIMARY KEY (node_id)
                """,
                """
                CREATE TABLE Relationships (
                    relationship_id STRING(36) NOT NULL,
                    source_node_id STRING(36) NOT NULL,
                    target_node_id STRING(36) NOT NULL,
                    relationship_type STRING(50) NOT NULL,
                    properties JSON,
                    confidence_score FLOAT64,
                    created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                    FOREIGN KEY (source_node_id) REFERENCES Nodes (node_id),
                    FOREIGN KEY (target_node_id) REFERENCES Nodes (node_id)
                ) PRIMARY KEY (relationship_id)
                """,
                """
                CREATE TABLE TextChunks (
                    chunk_id STRING(36) NOT NULL,
                    document_id STRING(36) NOT NULL,
                    chunk_index INT64 NOT NULL,
                    content STRING(MAX) NOT NULL,
                    embedding ARRAY<FLOAT64>,
                    metadata JSON,
                    created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
                    FOREIGN KEY (document_id) REFERENCES Documents (document_id)
                ) PRIMARY KEY (chunk_id)
                """
            ]
            
            # Execute drop statements
            for statement in drop_statements:
                try:
                    operation = database.update_ddl([statement])
                    operation.result(timeout=120)
                    logger.info(f"Executed: {statement}")
                except Exception as e:
                    logger.info(f"Ignoring drop error (table might not exist): {e}")
            
            # Execute create statements
            operation = database.update_ddl(create_statements)
            operation.result(timeout=300)
            logger.info("Successfully recreated all tables with correct schema")
            
        except Exception as e:
            logger.error(f"Error forcing schema recreation: {e}")
            raise

    def _get_document_ids(self, database) -> List[str]:
        """Get all existing document IDs"""
        try:
            with database.snapshot() as snapshot:
                results = snapshot.execute_sql("SELECT document_id FROM Documents")
                return [row[0] for row in results]
        except Exception as e:
            logger.warning(f"Could not get document IDs: {e}")
            return []
    
    async def _ensure_tables_exist(self, database):
        """Check if tables exist and create them if they don't"""
        try:
            # Get list of existing tables
            with database.snapshot() as snapshot:
                try:
                    # Try to query information schema to check if tables exist
                    results = snapshot.execute_sql(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = ''"
                    )
                    existing_tables = {row[0] for row in results}
                    
                    required_tables = {'Nodes', 'Relationships', 'Documents', 'TextChunks'}
                    missing_tables = required_tables - existing_tables
                    
                    if missing_tables:
                        logger.info(f"Missing tables: {missing_tables}. Creating only missing tables...")
                        await self._create_missing_tables(database, missing_tables)
                        # Verify schema after creating tables
                        await self._verify_table_schema(database)
                    else:
                        logger.info("All required tables exist")
                        # Always verify schema even if tables exist - force recreation if needed
                        logger.info("Forcing schema verification and recreation if needed...")
                        await self._force_schema_recreation(database)
                        
                except Exception as e:
                    logger.warning(f"Could not check existing tables: {e}. Skipping schema creation...")
                    # If we can't check, assume tables exist to avoid duplicate errors
                    
        except Exception as e:
            logger.warning(f"Error checking tables: {e}. Continuing anyway...")
            # Don't raise here, as the database might work even if we can't check tables
    
    async def store_graph(self, graph, documents: List, 
                         embeddings: List[List[float]]) -> bool:
        """Store the complete knowledge graph in Spanner using batch processing"""
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info(f"Storing knowledge graph with {len(documents)} documents in batches of {self.config.batch_size}")
            
            # Store documents in batches
            docs_success = await self._store_documents_batched(documents)
            
            # Store text chunks with embeddings in batches
            chunks_success = await self._store_text_chunks_batched(documents, embeddings)
            
            # Store graph nodes and relationships in batches (only if it's a NetworkX graph)
            nodes_success = True
            relationships_success = True
            if hasattr(graph, 'nodes') and hasattr(graph, 'edges'):
                nodes_success = await self._store_nodes_batched(graph)
                relationships_success = await self._store_relationships_batched(graph)
            else:
                logger.info("Graph is not NetworkX format, skipping node/relationship storage")
            
            overall_success = docs_success and chunks_success and nodes_success and relationships_success
            
            if overall_success:
                logger.info("✅ Knowledge graph stored successfully in Spanner")
            else:
                logger.warning("⚠️ Some parts of the knowledge graph failed to store")
            
            return overall_success
            
        except Exception as e:
            logger.error(f"❌ Error storing graph in Spanner: {e}")
            return False
    
    async def _store_documents_batched(self, documents: List) -> bool:
        """Store document metadata in Spanner using batched transactions"""
        try:
            batch_size = self.config.batch_size
            total_batches = (len(documents) + batch_size - 1) // batch_size
            successful_batches = 0
            
            # Track processed content hashes to avoid duplicates
            processed_hashes = set()
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Storing documents batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                # Retry logic for each batch
                for attempt in range(self.config.max_retries):
                    try:
                        def store_documents_batch(transaction):
                            batch_values = []
                            for doc in batch:
                                # Handle both Document objects and dictionaries
                                if hasattr(doc, 'page_content'):
                                    content = doc.page_content
                                    metadata = doc.metadata
                                elif isinstance(doc, dict) and 'content' in doc:
                                    content = doc['content']
                                    metadata = doc.get('metadata', {})
                                elif isinstance(doc, dict):
                                    content = str(doc)
                                    metadata = doc.get('metadata', {})
                                else:
                                    content = str(doc)
                                    metadata = {}
                                
                                content_hash = metadata.get('content_hash', hashlib.md5(content.encode()).hexdigest())
                                
                                # Skip duplicates within this batch and across batches
                                if content_hash in processed_hashes:
                                    logger.debug(f"Skipping duplicate document with hash {content_hash}")
                                    continue
                                
                                processed_hashes.add(content_hash)
                                document_id = str(uuid.uuid4())
                                
                                batch_values.append((
                                    document_id,
                                    metadata.get('source_type', 'unknown'),
                                    metadata.get('title', ''),
                                    content,
                                    json.dumps(metadata),
                                    content_hash,
                                    spanner.COMMIT_TIMESTAMP,
                                    spanner.COMMIT_TIMESTAMP
                                ))
                            
                            if batch_values:
                                transaction.insert(
                                    table="Documents",
                                    columns=["document_id", "source_type", "title", "content", 
                                            "metadata", "content_hash", "created_at", "updated_at"],
                                    values=batch_values
                                )
                        
                        self.database.run_in_transaction(store_documents_batch)
                        successful_batches += 1
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if attempt == self.config.max_retries - 1:
                            logger.error(f"Failed to store documents batch {batch_num} after {self.config.max_retries} attempts: {e}")
                        else:
                            logger.warning(f"Attempt {attempt + 1} failed for documents batch {batch_num}: {e}. Retrying...")
                            await asyncio.sleep(1)  # Brief delay before retry
            
            logger.info(f"Documents storage: {successful_batches}/{total_batches} batches successful")
            return successful_batches > 0  # Return True if at least one batch succeeded
            
        except Exception as e:
            logger.error(f"Error in batched document storage: {e}")
            return False
    
    async def _store_text_chunks_batched(self, documents: List, 
                                       embeddings: List[List[float]]) -> bool:
        """Store text chunks with embeddings in Spanner using batched transactions"""
        try:
            batch_size = self.config.batch_size
            total_batches = (len(documents) + batch_size - 1) // batch_size
            successful_batches = 0
            
            # Track processed chunk IDs to avoid duplicates
            processed_chunk_ids = set()
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size] if i < len(embeddings) else []
                batch_num = i // batch_size + 1
                
                logger.info(f"Storing text chunks batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
                # Retry logic for each batch
                for attempt in range(self.config.max_retries):
                    try:
                        def store_chunks_batch(transaction):
                            batch_values = []
                            for j, doc in enumerate(batch):
                                # Handle both Document objects and dictionaries
                                if hasattr(doc, 'page_content'):
                                    content = doc.page_content
                                    metadata = doc.metadata
                                elif isinstance(doc, dict) and 'content' in doc:
                                    content = doc['content']
                                    metadata = doc.get('metadata', {})
                                elif isinstance(doc, dict):
                                    content = str(doc)
                                    metadata = doc.get('metadata', {})
                                else:
                                    content = str(doc)
                                    metadata = {}
                                
                                # Generate unique chunk ID based on content hash and global index
                                content_hash = hashlib.md5(content.encode()).hexdigest()
                                global_index = i + j
                                chunk_id = metadata.get('chunk_id', f"{content_hash}_{global_index}")
                                
                                # Skip duplicates
                                if chunk_id in processed_chunk_ids:
                                    logger.debug(f"Skipping duplicate chunk {chunk_id}")
                                    continue
                                
                                processed_chunk_ids.add(chunk_id)
                                
                                document_id = metadata.get('document_id', str(uuid.uuid4()))
                                embedding = batch_embeddings[j] if j < len(batch_embeddings) else ([0.0] * 768)
                                
                                batch_values.append((
                                    chunk_id,
                                    document_id,
                                    metadata.get('chunk_index', global_index),
                                    content,
                                    embedding,
                                    json.dumps(metadata),
                                    spanner.COMMIT_TIMESTAMP
                                ))
                            
                            if batch_values:
                                transaction.insert(
                                    table="TextChunks",
                                    columns=["chunk_id", "document_id", "chunk_index", "content",
                                            "embedding", "metadata", "created_at"],
                                    values=batch_values
                                )
                        
                        self.database.run_in_transaction(store_chunks_batch)
                        successful_batches += 1
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if attempt == self.config.max_retries - 1:
                            logger.error(f"Failed to store chunks batch {batch_num} after {self.config.max_retries} attempts: {e}")
                        else:
                            logger.warning(f"Attempt {attempt + 1} failed for chunks batch {batch_num}: {e}. Retrying...")
                            await asyncio.sleep(1)  # Brief delay before retry
            
            logger.info(f"Text chunks storage: {successful_batches}/{total_batches} batches successful")
            return successful_batches > 0  # Return True if at least one batch succeeded
            
        except Exception as e:
            logger.error(f"Error in batched text chunks storage: {e}")
            return False
    
    async def _store_nodes_batched(self, graph) -> bool:
        """Store graph nodes in Spanner using batched transactions"""
        try:
            nodes = list(graph.nodes(data=True))
            if not nodes:
                logger.info("No nodes to store")
                return True
            
            batch_size = self.config.batch_size
            total_batches = (len(nodes) + batch_size - 1) // batch_size
            successful_batches = 0
            
            # Track processed node IDs to avoid duplicates
            processed_node_ids = set()
            
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Storing nodes batch {batch_num}/{total_batches} ({len(batch)} nodes)")
                
                # Retry logic for each batch
                for attempt in range(self.config.max_retries):
                    try:
                        def store_nodes_batch(transaction):
                            batch_values = []
                            for node_id, node_data in batch:
                                node_id_str = str(node_id)
                                
                                # Skip duplicates
                                if node_id_str in processed_node_ids:
                                    logger.debug(f"Skipping duplicate node {node_id_str}")
                                    continue
                                
                                processed_node_ids.add(node_id_str)
                                
                                batch_values.append((
                                    node_id_str,
                                    node_data.get('type', 'Entity'),
                                    node_data.get('name', str(node_id)),
                                    json.dumps(node_data),
                                    node_data.get('embedding', [0.0] * 768),
                                    node_data.get('source_document_id', ''),
                                    spanner.COMMIT_TIMESTAMP,
                                    spanner.COMMIT_TIMESTAMP
                                ))
                            
                            if batch_values:
                                transaction.insert(
                                    table="Nodes",
                                    columns=["node_id", "node_type", "name", "properties",
                                            "embedding", "source_document_id", "created_at", "updated_at"],
                                    values=batch_values
                                )
                        
                        self.database.run_in_transaction(store_nodes_batch)
                        successful_batches += 1
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if attempt == self.config.max_retries - 1:
                            logger.error(f"Failed to store nodes batch {batch_num} after {self.config.max_retries} attempts: {e}")
                        else:
                            logger.warning(f"Attempt {attempt + 1} failed for nodes batch {batch_num}: {e}. Retrying...")
                            await asyncio.sleep(1)  # Brief delay before retry
            
            logger.info(f"Nodes storage: {successful_batches}/{total_batches} batches successful")
            return successful_batches > 0  # Return True if at least one batch succeeded
            
        except Exception as e:
            logger.error(f"Error in batched nodes storage: {e}")
            return False
    
    async def _store_relationships_batched(self, graph) -> bool:
        """Store graph relationships in Spanner using batched transactions"""
        try:
            edges = list(graph.edges(data=True))
            if not edges:
                logger.info("No relationships to store")
                return True
            
            batch_size = self.config.batch_size
            total_batches = (len(edges) + batch_size - 1) // batch_size
            successful_batches = 0
            
            # Track processed relationship combinations to avoid duplicates
            processed_relationships = set()
            
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Storing relationships batch {batch_num}/{total_batches} ({len(batch)} relationships)")
                
                # Retry logic for each batch
                for attempt in range(self.config.max_retries):
                    try:
                        def store_relationships_batch(transaction):
                            batch_values = []
                            for source, target, edge_data in batch:
                                source_str = str(source)
                                target_str = str(target)
                                rel_type = edge_data.get('type', 'RELATED_TO')
                                
                                # Create unique key for this relationship
                                rel_key = (source_str, target_str, rel_type)
                                
                                # Skip duplicates
                                if rel_key in processed_relationships:
                                    logger.debug(f"Skipping duplicate relationship {rel_key}")
                                    continue
                                
                                processed_relationships.add(rel_key)
                                relationship_id = str(uuid.uuid4())
                                
                                batch_values.append((
                                    relationship_id,
                                    source_str,
                                    target_str,
                                    rel_type,
                                    json.dumps(edge_data),
                                    edge_data.get('confidence', 0.5),
                                    spanner.COMMIT_TIMESTAMP
                                ))
                            
                            if batch_values:
                                transaction.insert(
                                    table="Relationships",
                                    columns=["relationship_id", "source_node_id", "target_node_id",
                                            "relationship_type", "properties", "confidence_score", "created_at"],
                                    values=batch_values
                                )
                        
                        self.database.run_in_transaction(store_relationships_batch)
                        successful_batches += 1
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if attempt == self.config.max_retries - 1:
                            logger.error(f"Failed to store relationships batch {batch_num} after {self.config.max_retries} attempts: {e}")
                        else:
                            logger.warning(f"Attempt {attempt + 1} failed for relationships batch {batch_num}: {e}. Retrying...")
                            await asyncio.sleep(1)  # Brief delay before retry
            
            logger.info(f"Relationships storage: {successful_batches}/{total_batches} batches successful")
            return successful_batches > 0  # Return True if at least one batch succeeded
            
        except Exception as e:
            logger.error(f"Error in batched relationships storage: {e}")
            return False

class TrellixKnowledgeGraphBuilder:
    """Main orchestrator class for building the Trellix knowledge graph"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.data_sources: List[DataSource] = []
        self.document_processor = DocumentProcessor(config)
        self.graph_builder = VertexAIGraphBuilder(config)
        # Skip graph store initialization for vector-only RAG
        # self.graph_store = SpannerGraphStore(config)  # COMMENTED OUT
        self.vector_db = SpannerVectorDatabase(config)  # Use Spanner for vectors
        
        # Initialize data sources
        self._initialize_data_sources()
    
    def _initialize_data_sources(self):
        """Initialize all data sources"""
        # Confluence data source
        self.data_sources.append(ConfluenceDataSource(self.config))
        
        # Website data source
        self.data_sources.append(WebsiteDataSource(self.config))
        
        # Add other data sources as needed
    
    async def build_knowledge_graph(self) -> bool:
        """Main method to build the complete knowledge graph"""
        try:
            logger.info("🚀 Starting Trellix Knowledge Graph construction...")
            
            # Check dependencies first
            self._check_dependencies()
            
            # Step 1: Extract data from all sources
            logger.info("📚 Step 1: Extracting data from sources...")
            all_documents = await self._extract_all_data()
            
            if not all_documents:
                logger.warning("No documents extracted. Creating sample data for demonstration.")
                all_documents = self._create_sample_documents()
            
            # Step 2: Process and segment documents
            logger.info("🔧 Step 2: Processing and segmenting documents...")
            processed_documents = await self.document_processor.process_documents(all_documents)
            
            # Step 3: Initialize AI services
            logger.info("🧠 Step 3: Initializing AI services...")
            await self.graph_builder.initialize()
            
            # Initialize Vector Database for RAG
            if self.config.use_vector_db:
                logger.info("�️  Step 3b: Initializing Spanner Vector Database...")
                await self.vector_db.initialize()
            
            # SKIP traditional Spanner graph store initialization for vector-only RAG
            logger.info("📋 Skipping Spanner graph store initialization - Vector RAG mode only")
            
            # Step 4: COMMENTED OUT - Build knowledge graph (replaced with vector-only RAG)
            # logger.info("📊 Step 4: Building knowledge graph...")
            # knowledge_graph = await self.graph_builder.build_graph(processed_documents)
            logger.info("📊 Step 4: Graph creation skipped - Using vector-only RAG solution")
            knowledge_graph = None  # Skip graph creation
            
            # Step 5: Generate embeddings
            logger.info("🎯 Step 5: Generating embeddings for RAG...")
            texts = []
            for doc in processed_documents:
                print(doc,"------")
                if hasattr(doc, 'page_content'):
                    texts.append(doc.page_content)
                elif isinstance(doc, dict) and 'content' in doc:
                    texts.append(doc['content'])
                else:
                    texts.append(str(doc))
            
            embeddings = await self.graph_builder.generate_embeddings(texts)
            
            # Step 6: Store in Vector Database for RAG
            if self.config.use_vector_db and self.vector_db.initialized:
                logger.info("💾 Step 6: Storing documents in Spanner Vector Database for RAG...")
                vector_success = await self.vector_db.store_documents(processed_documents, embeddings)
                
                if vector_success:
                    # Log vector database statistics
                    db_info = self.vector_db.get_collection_info()
                    logger.info(f"✅ Vector Database Status: {db_info}")
                    success = True
                else:
                    logger.warning("⚠️ Vector database storage failed")
                    success = False
            else:
                logger.info("📋 Step 6: Vector database storage skipped")
                success = True
            
            # SKIP optional Spanner graph storage - Vector RAG mode only
            logger.info("� Spanner graph storage skipped - Vector RAG mode only")
            
            # Log statistics
            self._log_statistics(knowledge_graph, processed_documents)
            
            if self.config.use_vector_db:
                logger.info("✅ Trellix RAG Vector Database built successfully!")
            else:
                logger.info("✅ Trellix Knowledge processing completed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error building knowledge graph: {e}")
            return False
    
    def _check_dependencies(self):
        """Check and log available dependencies"""
        deps = {
            'Vertex AI': VERTEX_AI_AVAILABLE,
            'LangChain Basic': LANGCHAIN_BASIC_AVAILABLE,
            'LangChain Advanced': LANGCHAIN_ADVANCED_AVAILABLE,
            'NetworkX': NETWORKX_AVAILABLE,
            'Spanner Vector': SPANNER_VECTOR_AVAILABLE,
            'Confluence API': CONFLUENCE_AVAILABLE,
            'Web Scraping': WEB_SCRAPING_AVAILABLE,
            'spaCy': SPACY_AVAILABLE
        }
        
        logger.info("📦 Dependency Status:")
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            logger.info(f"   {status} {dep}")
        
        if not any(deps.values()):
            logger.warning("⚠️ Running with minimal dependencies - some features may be limited")
    
    def _create_sample_documents(self):
        """Create sample documents for demonstration when no data sources are available"""
        sample_docs = [
            {
                'content': "Trellix Endpoint Security provides comprehensive protection against malware, ransomware, and advanced threats. It includes real-time scanning, behavioral analysis, and machine learning capabilities.",
                'metadata': {'title': 'Endpoint Security Overview', 'source': 'sample', 'type': 'product_doc'}
            },
            {
                'content': "The SIEM solution offers centralized security monitoring, threat detection, and incident response capabilities. It integrates with multiple data sources and provides advanced analytics.",
                'metadata': {'title': 'SIEM Features', 'source': 'sample', 'type': 'product_doc'}
            },
            {
                'content': "Network Security solutions include firewall management, intrusion detection, and network segmentation capabilities for enterprise environments.",
                'metadata': {'title': 'Network Security', 'source': 'sample', 'type': 'product_doc'}
            }
        ]
        
        # Convert to Document objects if LangChain is available
        if LANGCHAIN_BASIC_AVAILABLE:
            from langchain.schema import Document
            return [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in sample_docs]
        else:
            return sample_docs
    
    async def _extract_all_data(self) -> List[Document]:
        """Extract data from all configured sources"""
        all_documents = []
        
        for data_source in self.data_sources:
            try:
                logger.info(f"Extracting data from {data_source.source_type}")
                documents = await data_source.extract_data()
                all_documents.extend(documents)
                logger.info(f"Extracted {len(documents)} documents from {data_source.source_type}")
            except Exception as e:
                logger.error(f"Error extracting from {data_source.source_type}: {e}")
        
        logger.info(f"Total documents extracted: {len(all_documents)}")
        return all_documents
    
    async def query_rag(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Query the RAG system for relevant documents
        
        Args:
            query: The search query
            n_results: Number of similar documents to return
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            if not self.config.use_vector_db or not self.vector_db.initialized:
                logger.error("Vector database not available for RAG queries")
                return []
            
            # Generate embedding for query if needed
            query_embedding = None
            if self.graph_builder.initialized and self.graph_builder.embedding_model:
                query_embeddings = await self.graph_builder.generate_embeddings([query])
                query_embedding = query_embeddings[0] if query_embeddings else None
            
            # Search vector database
            similar_docs = self.vector_db.search_similar(
                query=query, 
                n_results=n_results, 
                query_embedding=query_embedding
            )
            
            logger.info(f"Found {len(similar_docs)} relevant documents for query: '{query}'")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return []
    
    def get_rag_context(self, query: str, n_results: int = 3) -> str:
        """
        Get formatted context for RAG prompts
        
        Args:
            query: The search query
            n_results: Number of documents to include in context
            
        Returns:
            Formatted context string for LLM prompts
        """
        try:
            import asyncio
            similar_docs = asyncio.run(self.query_rag(query, n_results))
            
            if not similar_docs:
                return "No relevant context found."
            
            context_parts = []
            for i, doc in enumerate(similar_docs, 1):
                context_parts.append(f"[Context {i}]\n{doc['text']}\n")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting RAG context: {e}")
            return "Error retrieving context."
    
    def _log_statistics(self, graph, documents: List):
        """Log build statistics"""
        stats = {
            'total_documents': len(documents),
            'total_nodes': 0,
            'total_edges': 0,
            'node_types': {},
            'relationship_types': {}
        }
        
        # Handle case where graph might not be a NetworkX graph or is None (vector-only mode)
        try:
            if graph is None:
                stats['mode'] = 'vector_only_rag'
                stats['graph_type'] = 'None (Vector-only RAG)'
                logger.info("Running in vector-only RAG mode, no graph statistics available")
            elif hasattr(graph, 'number_of_nodes') and hasattr(graph, 'number_of_edges'):
                stats['total_nodes'] = graph.number_of_nodes()
                stats['total_edges'] = graph.number_of_edges()
                
                # Count node types
                if hasattr(graph, 'nodes'):
                    for _, node_data in graph.nodes(data=True):
                        node_type = node_data.get('type', 'Unknown')
                        stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
                
                # Count relationship types
                if hasattr(graph, 'edges'):
                    for _, _, edge_data in graph.edges(data=True):
                        rel_type = edge_data.get('type', 'Unknown')
                        stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1
            else:
                stats['graph_type'] = type(graph).__name__
                logger.info("Graph is not a NetworkX graph, limited statistics available")
        except Exception as e:
            logger.warning(f"Error collecting graph statistics: {e}")
            stats['error'] = str(e)
        
        logger.info(f"📊 Knowledge Graph Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict) and value:
                logger.info(f"   {key}: {dict(list(value.items())[:5])}")  # Show first 5 items
            else:
                logger.info(f"   {key}: {value}")

# Example usage and configuration
async def main():
    """Main function to demonstrate RAG system usage"""
    
    # Load configuration for Spanner-based RAG approach
    config = KnowledgeGraphConfig(
        # Google Cloud Spanner configuration
        project_id="svc-hackathon-prod15",
        spanner_instance_id="trellix-knowledge-graph",
        spanner_database_id="knowledge_graph_db",
        credentials_path="svc-hackathon-prod15-534bc641841c.json",
        
        # Enable Spanner vector database for RAG
        use_vector_db=True,
        vector_table_name="trellix_document_embeddings",
        vector_collection_name="trellix_knowledge",
        
        # Configure data sources
        confluence_spaces=["XDR","CSEH"],
        trellix_domains=[
            "https://docs.trellix.com"
        ],
        
        # Optimize for RAG performance
        chunk_size=400,
        chunk_overlap=50
    )
    
    # Build RAG system
    builder = TrellixKnowledgeGraphBuilder(config)
    success = await builder.build_knowledge_graph()
    
    if success:
        print("✅ RAG system built successfully!")
        
        # Demo RAG query
        query = "What are the main features of Trellix endpoint security?"
        relevant_docs = await builder.query_rag(query, n_results=3)
        print(f"\n🔍 RAG Query Result for: '{query}'")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"[{i}] {doc['text'][:150]}...")
    else:
        print("❌ Failed to build RAG system")

if __name__ == "__main__":
    asyncio.run(main())