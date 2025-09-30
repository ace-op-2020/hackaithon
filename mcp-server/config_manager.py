"""
Configuration Management for Trellix Knowledge Graph
==================================================

Centralized configuration management with environment variable support
and validation for the knowledge graph builder.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

@dataclass
class ConfluenceConfig:
    """Configuration for Confluence data source"""
    url: str = ""
    username: str = ""
    api_token: str = ""
    spaces: List[str] = field(default_factory=list)
    page_limit: int = 1000
    include_attachments: bool = False
    
    def is_configured(self) -> bool:
        """Check if Confluence is properly configured"""
        return bool(self.url and self.username and self.api_token and self.spaces)

@dataclass
class WebScrapingConfig:
    """Configuration for web scraping"""
    domains: List[str] = field(default_factory=list)
    max_pages_per_domain: int = 100
    request_delay: float = 1.0
    timeout: int = 30
    user_agent: str = "Trellix Knowledge Graph Builder 1.0"
    respect_robots_txt: bool = True
    allowed_content_types: List[str] = field(default_factory=lambda: [
        "text/html", "text/plain", "application/pdf"
    ])

@dataclass
class VertexAIConfig:
    """Configuration for Vertex AI services"""
    project_id: str = "svc-hackathon-prod15"
    location: str = "us-central1"
    model_name: str = "gemini-2.0-flash-lite-001"
    embedding_model: str = "textembedding-gecko@003"
    max_tokens: int = 8192
    temperature: float = 0.1
    top_p: float = 0.8
    top_k: int = 40
    
    # Rate limiting
    requests_per_minute: int = 60
    batch_size: int = 10

@dataclass
class SpannerConfig:
    """Configuration for Spanner Graph database"""
    instance_id: str = "trellix-knowledge-graph"
    database_id: str = "knowledge_graph_db"
    max_sessions: int = 100
    timeout: int = 300
    
    # Performance settings
    batch_size: int = 1000
    max_mutations_per_batch: int = 20000

@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 8192
    min_chunk_size: int = 100
    
    # Text cleaning
    remove_html: bool = True
    remove_urls: bool = False
    normalize_whitespace: bool = True
    
    # Language processing
    language: str = "en"
    enable_ner: bool = True
    enable_pos_tagging: bool = False

@dataclass
class GraphConfig:
    """Configuration for knowledge graph construction"""
    max_nodes: int = 50000
    max_relationships: int = 100000
    similarity_threshold: float = 0.8
    confidence_threshold: float = 0.5
    
    # Node types allowed in the graph
    allowed_node_types: List[str] = field(default_factory=lambda: [
        "Product", "Feature", "Component", "Process", "Person", 
        "Organization", "Document", "Concept", "Technology", "Location"
    ])
    
    # Relationship types allowed in the graph
    allowed_relationship_types: List[str] = field(default_factory=lambda: [
        "USES", "IMPLEMENTS", "RELATES_TO", "PART_OF", "MANAGES", 
        "DEPENDS_ON", "CONTAINS", "CREATES", "SUPPORTS", "INTEGRATES_WITH"
    ])
    
    # Community detection
    enable_community_detection: bool = True
    community_algorithm: str = "louvain"  # louvain, leiden, greedy_modularity

@dataclass
class KnowledgeGraphConfig:
    """Main configuration class for the knowledge graph builder"""
    
    # Authentication
    credentials_path: str = ""
    
    # Component configurations
    confluence: ConfluenceConfig = field(default_factory=ConfluenceConfig)
    web_scraping: WebScrapingConfig = field(default_factory=WebScrapingConfig)
    vertex_ai: VertexAIConfig = field(default_factory=VertexAIConfig)
    spanner: SpannerConfig = field(default_factory=SpannerConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Output
    output_directory: str = "./output"
    enable_visualization: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "graphml", "html"])
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        self._setup_logging()
        self._create_output_directory()
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Check credentials
        if self.credentials_path and not os.path.exists(self.credentials_path):
            errors.append(f"Credentials file not found: {self.credentials_path}")
        
        # Validate Vertex AI config
        if not self.vertex_ai.project_id:
            errors.append("Vertex AI project_id is required")
        
        # Validate processing config
        if self.processing.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.processing.chunk_overlap >= self.processing.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")
        
        # Validate graph config
        if self.graph.similarity_threshold < 0 or self.graph.similarity_threshold > 1:
            errors.append("similarity_threshold must be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        logging_config = {
            'level': log_level,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
        
        if self.log_file:
            logging_config['filename'] = self.log_file
            logging_config['filemode'] = 'a'
        
        logging.basicConfig(**logging_config)
    
    def _create_output_directory(self):
        """Create output directory if it doesn't exist"""
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)

class ConfigurationManager:
    """Manage configuration loading from various sources"""
    
    @staticmethod
    def load_from_file(config_path: str) -> KnowledgeGraphConfig:
        """Load configuration from YAML or JSON file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config_data = json.load(f)
            else:
                raise ValueError("Configuration file must be YAML or JSON")
        
        return ConfigurationManager._create_config_from_dict(config_data)
    
    @staticmethod
    def load_from_environment() -> KnowledgeGraphConfig:
        """Load configuration from environment variables"""
        config = KnowledgeGraphConfig()
        
        # Update configuration from environment variables
        config.credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', config.credentials_path)
        config.log_level = os.getenv('LOG_LEVEL', config.log_level)
        config.output_directory = os.getenv('OUTPUT_DIRECTORY', config.output_directory)
        
        # Vertex AI configuration
        config.vertex_ai.project_id = os.getenv('VERTEX_AI_PROJECT_ID', config.vertex_ai.project_id)
        config.vertex_ai.location = os.getenv('VERTEX_AI_LOCATION', config.vertex_ai.location)
        config.vertex_ai.model_name = os.getenv('VERTEX_AI_MODEL_NAME', config.vertex_ai.model_name)
        
        # Confluence configuration
        config.confluence.url = os.getenv('CONFLUENCE_URL', config.confluence.url)
        config.confluence.username = os.getenv('CONFLUENCE_USERNAME', config.confluence.username)
        config.confluence.api_token = os.getenv('CONFLUENCE_API_TOKEN', config.confluence.api_token)
        
        # Parse list environment variables
        if os.getenv('CONFLUENCE_SPACES'):
            config.confluence.spaces = os.getenv('CONFLUENCE_SPACES', '').split(',')
        
        if os.getenv('WEB_SCRAPING_DOMAINS'):
            config.web_scraping.domains = os.getenv('WEB_SCRAPING_DOMAINS', '').split(',')
        
        # Spanner configuration
        config.spanner.instance_id = os.getenv('SPANNER_INSTANCE_ID', config.spanner.instance_id)
        config.spanner.database_id = os.getenv('SPANNER_DATABASE_ID', config.spanner.database_id)
        
        # Numeric configurations
        if os.getenv('CHUNK_SIZE'):
            config.processing.chunk_size = int(os.getenv('CHUNK_SIZE'))
        
        if os.getenv('CHUNK_OVERLAP'):
            config.processing.chunk_overlap = int(os.getenv('CHUNK_OVERLAP'))
        
        return config
    
    @staticmethod
    def _create_config_from_dict(config_data: Dict[str, Any]) -> KnowledgeGraphConfig:
        """Create configuration object from dictionary"""
        config = KnowledgeGraphConfig()
        
        # Update main config
        for key, value in config_data.items():
            if hasattr(config, key) and not key.startswith('_'):
                if isinstance(getattr(config, key), (ConfluenceConfig, WebScrapingConfig, 
                                                   VertexAIConfig, SpannerConfig, 
                                                   ProcessingConfig, GraphConfig)):
                    # Handle nested configuration objects
                    nested_config = getattr(config, key)
                    if isinstance(value, dict):
                        for nested_key, nested_value in value.items():
                            if hasattr(nested_config, nested_key):
                                setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(config, key, value)
        
        return config
    
    @staticmethod
    def save_to_file(config: KnowledgeGraphConfig, output_path: str):
        """Save configuration to file"""
        config_dict = ConfigurationManager._config_to_dict(config)
        
        with open(output_path, 'w') as f:
            if output_path.endswith('.yaml') or output_path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif output_path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError("Output file must be YAML or JSON")
        
        logger.info(f"Configuration saved to {output_path}")
    
    @staticmethod
    def _config_to_dict(config: KnowledgeGraphConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        result = {}
        
        for key, value in config.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, (ConfluenceConfig, WebScrapingConfig, 
                                    VertexAIConfig, SpannerConfig, 
                                    ProcessingConfig, GraphConfig)):
                    result[key] = {k: v for k, v in value.__dict__.items() if not k.startswith('_')}
                else:
                    result[key] = value
        
        return result

def create_sample_config() -> str:
    """Create a sample configuration file"""
    sample_config = {
        "credentials_path": "./credentials.json",
        "log_level": "INFO",
        "output_directory": "./output",
        "confluence": {
            "url": "https://your-company.atlassian.net/wiki",
            "username": "your-email@company.com",
            "api_token": "your-api-token",
            "spaces": ["PROD", "DOCS", "HELP"],
            "page_limit": 1000
        },
        "web_scraping": {
            "domains": [
                "https://www.trellix.com",
                "https://docs.trellix.com"
            ],
            "max_pages_per_domain": 100,
            "request_delay": 1.0
        },
        "vertex_ai": {
            "project_id": "your-project-id",
            "location": "us-central1",
            "model_name": "gemini-2.0-flash-lite-001",
            "embedding_model": "textembedding-gecko@003"
        },
        "spanner": {
            "instance_id": "trellix-knowledge-graph",
            "database_id": "knowledge_graph_db"
        },
        "processing": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "language": "en"
        },
        "graph": {
            "max_nodes": 50000,
            "similarity_threshold": 0.8,
            "confidence_threshold": 0.5
        }
    }
    
    output_path = "config_sample.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    return output_path

# Default configuration for quick setup
def get_default_config() -> KnowledgeGraphConfig:
    """Get default configuration for quick setup"""
    config = KnowledgeGraphConfig()
    
    # Set some sensible defaults for Trellix
    config.web_scraping.domains = [
        "https://www.trellix.com",
        "https://docs.trellix.com",
        "https://community.trellix.com"
    ]
    
    config.processing.chunk_size = 1000
    config.processing.chunk_overlap = 200
    
    config.graph.max_nodes = 50000
    config.graph.similarity_threshold = 0.8
    
    # Try to load from environment
    try:
        env_config = ConfigurationManager.load_from_environment()
        # Merge environment config with defaults
        for key, value in env_config.__dict__.items():
            if not key.startswith('_') and value:
                setattr(config, key, value)
    except Exception as e:
        logger.warning(f"Could not load environment configuration: {e}")
    
    return config

if __name__ == "__main__":
    # Create sample configuration file
    sample_path = create_sample_config()
    print(f"Sample configuration created at: {sample_path}")
    
    # Test loading configuration
    try:
        config = ConfigurationManager.load_from_file(sample_path)
        print("Configuration loaded successfully!")
        print(f"Vertex AI Project: {config.vertex_ai.project_id}")
        print(f"Output Directory: {config.output_directory}")
    except Exception as e:
        print(f"Error loading configuration: {e}")