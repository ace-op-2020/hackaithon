"""
Minimal Trellix Knowledge Graph Builder
=====================================

A simplified version of the knowledge graph builder that works with 
basic dependencies for demonstration purposes.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import uuid

# Core dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Google Cloud dependencies (optional for demo)
try:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel
    from vertexai.language_models import TextEmbeddingModel
    HAS_VERTEX_AI = True
except ImportError:
    HAS_VERTEX_AI = False

try:
    from google.cloud import spanner
    HAS_SPANNER = True
except ImportError:
    HAS_SPANNER = False

# LangChain dependencies (optional for demo)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    # Create a simple Document class for demo
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SimpleConfig:
    """Simplified configuration for demo"""
    project_id: str = "svc-hackathon-prod15"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_nodes: int = 10000

class SimpleTextSplitter:
    """Simple text splitter when LangChain is not available"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundaries
            if end < len(text):
                # Look for a space to break on
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks

class SampleDataSource:
    """Sample data source with Trellix product information"""
    
    def __init__(self):
        self.documents = []
    
    async def extract_data(self) -> List[Document]:
        """Generate sample Trellix product data"""
        sample_documents = [
            Document(
                page_content="""
                Trellix Endpoint Security Overview
                
                Trellix Endpoint Security provides comprehensive protection against advanced threats.
                The solution includes real-time scanning, behavioral analysis, and machine learning
                capabilities to detect and prevent malware, ransomware, and zero-day attacks.
                
                Key Features:
                - Real-time threat detection using advanced heuristics
                - Centralized management console for enterprise deployment
                - Integration with SIEM systems for enhanced monitoring
                - Automatic threat response and remediation
                - Cloud-based threat intelligence feeds
                - Machine learning algorithms for behavioral analysis
                
                The antivirus engine combines signature-based detection with behavioral analysis
                to identify both known and unknown threats. The solution supports Windows, Mac,
                and Linux endpoints with a unified management interface.
                
                Installation Requirements:
                - Windows 10/11 or Windows Server 2016+
                - Minimum 4GB RAM, 2GB disk space
                - Administrative privileges required
                - Network connectivity for updates
                """,
                metadata={
                    'title': 'Trellix Endpoint Security Overview',
                    'product': 'Endpoint Security',
                    'category': 'Security Solutions',
                    'type': 'product_overview'
                }
            ),
            Document(
                page_content="""
                Trellix Network Security Platform
                
                Trellix Network Security provides network-based threat detection and prevention
                capabilities. The solution monitors network traffic in real-time to identify
                and block malicious activities, intrusions, and data exfiltration attempts.
                
                Core Capabilities:
                - Deep packet inspection with SSL/TLS decryption
                - Intrusion detection and prevention (IDS/IPS)
                - Network traffic analysis and anomaly detection
                - Advanced persistent threat (APT) detection
                - Data loss prevention (DLP) integration
                - Automated threat response actions
                
                The platform uses machine learning algorithms to establish network baselines
                and detect anomalous behavior patterns. Integration with endpoint security
                provides comprehensive threat visibility across the entire infrastructure.
                
                Deployment Options:
                - Physical appliances for high-performance environments
                - Virtual appliances for cloud and hybrid deployments
                - Software sensors for distributed monitoring
                - Cloud-native deployment for SaaS environments
                """,
                metadata={
                    'title': 'Trellix Network Security Platform',
                    'product': 'Network Security',
                    'category': 'Security Solutions',
                    'type': 'product_overview'
                }
            ),
            Document(
                page_content="""
                Trellix SIEM Integration Guide
                
                The Trellix Security Information and Event Management (SIEM) platform
                centralizes security event monitoring and incident response. It collects,
                correlates, and analyzes security events from multiple sources.
                
                Integration Features:
                - Real-time event correlation across multiple security tools
                - Customizable dashboards and reporting capabilities
                - Automated incident response playbooks
                - Threat hunting and forensic investigation tools
                - Compliance reporting for regulatory requirements
                - Machine learning-based anomaly detection
                
                The SIEM integrates with endpoint protection, network security, and
                external threat intelligence feeds to provide comprehensive security
                monitoring. It supports custom rule creation and automated workflows.
                
                Configuration Steps:
                1. Install SIEM connectors on security devices
                2. Configure log forwarding and collection
                3. Set up correlation rules and alerts
                4. Create custom dashboards and reports
                5. Configure automated response actions
                6. Test integration and validate data flow
                """,
                metadata={
                    'title': 'Trellix SIEM Integration Guide',
                    'product': 'SIEM',
                    'category': 'Security Management',
                    'type': 'integration_guide'
                }
            ),
            Document(
                page_content="""
                Troubleshooting Common Trellix Issues
                
                This guide covers common issues and solutions for Trellix security products.
                
                Endpoint Security Issues:
                
                Problem: Agent fails to start
                Solution: Check Windows services, verify permissions, restart ePO server
                
                Problem: Real-time scanning disabled
                Solution: Check policy settings, verify license, update signatures
                
                Problem: High CPU usage
                Solution: Adjust scan schedules, exclude system files, update client
                
                Network Security Issues:
                
                Problem: Dropped packets
                Solution: Check network capacity, adjust inspection rules, verify hardware
                
                Problem: SSL inspection fails
                Solution: Verify certificates, check cipher compatibility, update firmware
                
                Problem: False positives
                Solution: Tune detection rules, whitelist known good traffic, update signatures
                
                SIEM Issues:
                
                Problem: Missing events
                Solution: Check log forwarding, verify network connectivity, restart collectors
                
                Problem: Slow dashboard loading
                Solution: Optimize queries, archive old data, add database indexes
                
                Problem: Alert fatigue
                Solution: Tune correlation rules, prioritize critical alerts, group similar events
                """,
                metadata={
                    'title': 'Troubleshooting Common Trellix Issues',
                    'product': 'All Products',
                    'category': 'Troubleshooting',
                    'type': 'troubleshooting_guide'
                }
            ),
            Document(
                page_content="""
                Trellix API Documentation
                
                The Trellix platform provides REST APIs for automation and integration.
                
                Authentication:
                All API calls require authentication using API keys or OAuth tokens.
                Include the Authorization header: "Bearer <your-token>"
                
                Endpoint Security API:
                
                GET /api/v1/endpoints - List all managed endpoints
                POST /api/v1/endpoints/{id}/scan - Initiate on-demand scan
                GET /api/v1/threats - Retrieve threat detections
                PUT /api/v1/policies/{id} - Update security policies
                
                Network Security API:
                
                GET /api/v1/sensors - List network sensors
                GET /api/v1/alerts - Retrieve security alerts
                POST /api/v1/rules - Create detection rules
                DELETE /api/v1/rules/{id} - Remove detection rules
                
                SIEM API:
                
                GET /api/v1/events - Query security events
                POST /api/v1/incidents - Create security incidents
                GET /api/v1/dashboards - List available dashboards
                PUT /api/v1/rules/{id} - Update correlation rules
                
                Rate Limits:
                - 1000 requests per hour per API key
                - 10 requests per second burst limit
                - Use pagination for large result sets
                
                Error Handling:
                - 400: Bad Request - Invalid parameters
                - 401: Unauthorized - Invalid credentials
                - 403: Forbidden - Insufficient permissions
                - 429: Too Many Requests - Rate limit exceeded
                - 500: Internal Server Error - Contact support
                """,
                metadata={
                    'title': 'Trellix API Documentation',
                    'product': 'Platform APIs',
                    'category': 'API Reference',
                    'type': 'api_documentation'
                }
            )
        ]
        
        logger.info(f"Generated {len(sample_documents)} sample documents")
        self.documents = sample_documents
        return sample_documents

class SimpleDocumentProcessor:
    """Simple document processor"""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        if HAS_LANGCHAIN:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        else:
            self.text_splitter = SimpleTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
    
    async def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and segment documents into chunks"""
        processed_docs = []
        
        for doc in documents:
            try:
                # Clean text
                cleaned_content = self._clean_text(doc.page_content)
                
                # Split into chunks
                chunks = self.text_splitter.split_text(cleaned_content)
                
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 50:  # Filter short chunks
                        chunk_metadata = doc.metadata.copy()
                        chunk_metadata.update({
                            'chunk_id': f"{doc.metadata.get('title', 'doc')}_{i}",
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'chunk_size': len(chunk)
                        })
                        
                        processed_docs.append(Document(
                            page_content=chunk,
                            metadata=chunk_metadata
                        ))
            
            except Exception as e:
                logger.error(f"Error processing document: {e}")
        
        logger.info(f"Processed {len(documents)} documents into {len(processed_docs)} chunks")
        return processed_docs
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text.strip()

class SimpleKnowledgeGraphBuilder:
    """Simplified knowledge graph builder for demonstration"""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.data_source = SampleDataSource()
        self.document_processor = SimpleDocumentProcessor(config)
    
    async def build_knowledge_graph(self) -> bool:
        """Build a simple knowledge graph"""
        try:
            logger.info("🚀 Starting Trellix Knowledge Graph construction...")
            
            # Step 1: Extract sample data
            logger.info("📚 Extracting sample data...")
            documents = await self.data_source.extract_data()
            
            # Step 2: Process documents
            logger.info("🔧 Processing and segmenting documents...")
            processed_docs = await self.document_processor.process_documents(documents)
            
            # Step 3: Create simple knowledge graph
            logger.info("🧠 Creating knowledge graph...")
            graph = self._create_simple_graph(processed_docs)
            
            # Step 4: Analyze graph
            logger.info("📊 Analyzing knowledge graph...")
            self._analyze_graph(graph, processed_docs)
            
            logger.info("✅ Knowledge graph built successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return False
    
    def _create_simple_graph(self, documents: List[Document]) -> Dict[str, Any]:
        """Create a simple graph structure"""
        graph = {
            'nodes': {},
            'edges': [],
            'categories': set(),
            'products': set()
        }
        
        # Extract entities and relationships from documents
        for doc in documents:
            product = doc.metadata.get('product', 'Unknown')
            category = doc.metadata.get('category', 'Unknown')
            title = doc.metadata.get('title', 'Untitled')
            
            # Add product node
            if product != 'Unknown':
                graph['products'].add(product)
                if product not in graph['nodes']:
                    graph['nodes'][product] = {
                        'type': 'Product',
                        'name': product,
                        'documents': [],
                        'features': set()
                    }
                graph['nodes'][product]['documents'].append(title)
            
            # Add category node
            if category != 'Unknown':
                graph['categories'].add(category)
                if category not in graph['nodes']:
                    graph['nodes'][category] = {
                        'type': 'Category',
                        'name': category,
                        'products': set()
                    }
                
                # Link product to category
                if product != 'Unknown':
                    graph['nodes'][category]['products'].add(product)
                    graph['edges'].append({
                        'source': product,
                        'target': category,
                        'type': 'BELONGS_TO'
                    })
            
            # Extract features from content
            content_lower = doc.page_content.lower()
            features = self._extract_features(content_lower)
            
            for feature in features:
                if feature not in graph['nodes']:
                    graph['nodes'][feature] = {
                        'type': 'Feature',
                        'name': feature,
                        'products': set()
                    }
                
                if product != 'Unknown':
                    graph['nodes'][feature]['products'].add(product)
                    graph['nodes'][product]['features'].add(feature)
                    graph['edges'].append({
                        'source': product,
                        'target': feature,
                        'type': 'HAS_FEATURE'
                    })
        
        return graph
    
    def _extract_features(self, content: str) -> List[str]:
        """Extract feature keywords from content"""
        features = []
        
        # Common security features
        feature_keywords = [
            'real-time scanning', 'behavioral analysis', 'machine learning',
            'threat detection', 'intrusion prevention', 'endpoint protection',
            'network monitoring', 'ssl inspection', 'deep packet inspection',
            'threat intelligence', 'incident response', 'automated remediation',
            'centralized management', 'policy management', 'compliance reporting',
            'forensic investigation', 'threat hunting', 'anomaly detection',
            'signature-based detection', 'heuristic analysis', 'sandbox analysis'
        ]
        
        for keyword in feature_keywords:
            if keyword in content:
                features.append(keyword.title().replace(' ', '_'))
        
        return features
    
    def _analyze_graph(self, graph: Dict[str, Any], documents: List[Document]):
        """Analyze and display graph statistics"""
        print("\n" + "="*60)
        print("🧠 TRELLIX KNOWLEDGE GRAPH ANALYSIS")
        print("="*60)
        
        print(f"📊 Graph Statistics:")
        print(f"   • Total Nodes: {len(graph['nodes'])}")
        print(f"   • Total Edges: {len(graph['edges'])}")
        print(f"   • Products: {len(graph['products'])}")
        print(f"   • Categories: {len(graph['categories'])}")
        print(f"   • Documents Processed: {len(documents)}")
        
        print(f"\n🔧 Products Found:")
        for product in sorted(graph['products']):
            node = graph['nodes'][product]
            feature_count = len(node.get('features', []))
            doc_count = len(node.get('documents', []))
            print(f"   • {product}: {feature_count} features, {doc_count} documents")
        
        print(f"\n📂 Categories:")
        for category in sorted(graph['categories']):
            node = graph['nodes'][category]
            product_count = len(node.get('products', []))
            print(f"   • {category}: {product_count} products")
        
        # Show sample features
        feature_nodes = [n for n in graph['nodes'].values() if n['type'] == 'Feature']
        if feature_nodes:
            print(f"\n⚡ Top Features (showing first 10):")
            for i, feature in enumerate(sorted(feature_nodes, 
                                             key=lambda x: len(x.get('products', [])), 
                                             reverse=True)[:10]):
                product_count = len(feature.get('products', []))
                print(f"   {i+1:2d}. {feature['name']}: {product_count} products")
        
        # Show relationships
        relationship_types = {}
        for edge in graph['edges']:
            rel_type = edge['type']
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        if relationship_types:
            print(f"\n🔗 Relationship Types:")
            for rel_type, count in sorted(relationship_types.items()):
                print(f"   • {rel_type}: {count} relationships")
        
        # Sample queries
        print(f"\n🔍 Sample Knowledge Graph Queries:")
        print(f"   • 'What features does Endpoint Security have?'")
        self._query_product_features(graph, 'Endpoint Security')
        
        print(f"   • 'Which products have Real-time Scanning?'")
        self._query_feature_products(graph, 'Real-Time_Scanning')
        
        print(f"   • 'What products are in Security Solutions category?'")
        self._query_category_products(graph, 'Security Solutions')
    
    def _query_product_features(self, graph: Dict[str, Any], product: str):
        """Query features for a specific product"""
        if product in graph['nodes']:
            features = graph['nodes'][product].get('features', [])
            if features:
                print(f"     → {', '.join(sorted(features)[:5])}{'...' if len(features) > 5 else ''}")
            else:
                print(f"     → No features found")
        else:
            print(f"     → Product not found")
    
    def _query_feature_products(self, graph: Dict[str, Any], feature: str):
        """Query products that have a specific feature"""
        if feature in graph['nodes']:
            products = graph['nodes'][feature].get('products', [])
            if products:
                print(f"     → {', '.join(sorted(products))}")
            else:
                print(f"     → No products found")
        else:
            print(f"     → Feature not found")
    
    def _query_category_products(self, graph: Dict[str, Any], category: str):
        """Query products in a specific category"""
        if category in graph['nodes']:
            products = graph['nodes'][category].get('products', [])
            if products:
                print(f"     → {', '.join(sorted(products))}")
            else:
                print(f"     → No products found")
        else:
            print(f"     → Category not found")

async def main():
    """Main demonstration function"""
    print("🚀 Minimal Trellix Knowledge Graph Builder Demo")
    print("=" * 60)
    
    # Check available dependencies
    print("📦 Checking available dependencies:")
    print(f"   • Pandas: {'✅' if HAS_PANDAS else '❌'}")
    print(f"   • NetworkX: {'✅' if HAS_NETWORKX else '❌'}")
    print(f"   • LangChain: {'✅' if HAS_LANGCHAIN else '❌'}")
    print(f"   • Vertex AI: {'✅' if HAS_VERTEX_AI else '❌'}")
    print(f"   • Spanner: {'✅' if HAS_SPANNER else '❌'}")
    
    if not any([HAS_PANDAS, HAS_NETWORKX, HAS_LANGCHAIN]):
        print("\n⚠️  Running in minimal mode with basic Python libraries only")
    
    try:
        # Create configuration
        config = SimpleConfig()
        
        # Build knowledge graph
        builder = SimpleKnowledgeGraphBuilder(config)
        success = await builder.build_knowledge_graph()
        
        if success:
            print("\n" + "="*60)
            print("✅ DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("🎯 This demonstrates the core knowledge graph concepts:")
            print("   • Multi-source data extraction")
            print("   • Document processing and chunking")
            print("   • Entity and relationship extraction")
            print("   • Graph analysis and querying")
            print("   • Integration with chatbot systems")
            
            print("\n🔗 Next Steps:")
            print("   1. Install full dependencies: pip install -r requirements.txt")
            print("   2. Configure Google Cloud credentials")
            print("   3. Set up Confluence and web scraping")
            print("   4. Run the full knowledge_grapth_builder.py")
            print("   5. Integrate with your MCP server")
        else:
            print("\n❌ Demo failed")
    
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())