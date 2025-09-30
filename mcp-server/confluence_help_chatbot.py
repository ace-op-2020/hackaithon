#!/usr/bin/env python3
"""
Confluence Help Chatbot - RAG Implementation
============================================

Optimized configuration for a solution help chatbot using RAG on Confluence documentation.
This implementation uses the best practices for embedding and retrieving help content.

Features:
- Optimized for technical documentation and FAQ content
- Best embedding model for help desk scenarios
- Intelligent chunking for Q&A pairs
- Context-aware response generation

Usage:
    python confluence_help_chatbot.py
"""

import asyncio
import logging
from typing import List, Dict
from knowledge_grapth_builder import TrellixKnowledgeGraphBuilder, KnowledgeGraphConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfluenceHelpChatbot:
    """Specialized chatbot for Confluence help documentation"""
    
    def __init__(self):
        # Optimized configuration for help chatbot
        self.config = KnowledgeGraphConfig(
            # Google Cloud Spanner configuration
            project_id="svc-hackathon-prod15",
            spanner_instance_id="trellix-knowledge-graph", 
            spanner_database_id="knowledge_graph_db",
            credentials_path="svc-hackathon-prod15-534bc641841c.json",
            
            # Embedding optimization for help content
            embedding_model="text-embedding-004",  # Best for technical docs
            
            # Confluence spaces optimized for help content
            confluence_spaces=[
                "HELP",      # Help documentation
                "KB",        # Knowledge base  
                "FAQ",       # Frequently asked questions
                "SUPPORT",   # Support articles
                "DOCS",      # Product documentation
                "TROUBLESHOOTING"  # Troubleshooting guides
            ],
            
            # Text processing optimized for Q&A
            chunk_size=512,      # Optimal for help articles and FAQ entries
            chunk_overlap=64,    # Preserve context between chunks
            
            # Vector database settings
            use_vector_db=True,
            vector_table_name="help_chatbot_embeddings",
            vector_collection_name="trellix_help"
        )
        
        self.rag_builder = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the help chatbot RAG system"""
        try:
            logger.info("🤖 Initializing Confluence Help Chatbot...")
            
            self.rag_builder = TrellixKnowledgeGraphBuilder(self.config)
            
            # Build the knowledge base from Confluence
            success = await self.rag_builder.build_knowledge_graph()
            
            if success:
                self.initialized = True
                logger.info("✅ Help chatbot initialized successfully!")
                
                # Log knowledge base statistics
                db_info = self.rag_builder.vector_db.get_collection_info()
                logger.info(f"📚 Knowledge Base: {db_info.get('document_count', 0)} help articles indexed")
                
                return True
            else:
                logger.error("❌ Failed to initialize help chatbot")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            return False
    
    async def get_help_response(self, user_question: str, max_context_docs: int = 3) -> Dict:
        """
        Get a help response using RAG
        
        Args:
            user_question: The user's help question
            max_context_docs: Number of relevant documents to include in context
            
        Returns:
            Dictionary with response, sources, and confidence
        """
        try:
            if not self.initialized:
                return {"error": "Chatbot not initialized"}
            
            # Find relevant help articles
            relevant_docs = await self.rag_builder.query_rag(
                query=user_question, 
                n_results=max_context_docs
            )
            
            if not relevant_docs:
                return {
                    "response": "I couldn't find specific information about that topic. Please check our main documentation or contact support.",
                    "sources": [],
                    "confidence": "low"
                }
            
            # Get formatted context for LLM
            context = self.rag_builder.get_rag_context(user_question, n_results=max_context_docs)
            
            # Prepare response with context
            response_data = {
                "context": context,
                "relevant_docs": relevant_docs,
                "sources": [
                    {
                        "title": doc.get('metadata', {}).get('title', 'Untitled'),
                        "url": doc.get('metadata', {}).get('url', ''),
                        "similarity": doc.get('similarity', 0),
                        "snippet": doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                    }
                    for doc in relevant_docs
                ],
                "confidence": self._calculate_confidence(relevant_docs)
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error getting help response: {e}")
            return {"error": f"Error processing question: {str(e)}"}
    
    def _calculate_confidence(self, relevant_docs: List[Dict]) -> str:
        """Calculate confidence level based on similarity scores"""
        if not relevant_docs:
            return "none"
        
        avg_similarity = sum(doc.get('similarity', 0) for doc in relevant_docs) / len(relevant_docs)
        
        if avg_similarity > 0.8:
            return "high"
        elif avg_similarity > 0.6:
            return "medium"
        else:
            return "low"
    
    async def search_help_topics(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for help topics without full response generation"""
        if not self.initialized:
            return []
        
        return await self.rag_builder.query_rag(query, n_results=limit)

async def demo_help_chatbot():
    """Demonstrate the help chatbot functionality"""
    
    print("🤖 Confluence Help Chatbot Demo")
    print("=" * 50)
    
    chatbot = ConfluenceHelpChatbot()
    
    # Initialize
    success = await chatbot.initialize()
    if not success:
        print("❌ Failed to initialize chatbot")
        return
    
    # Sample help questions
    help_questions = [
        "How do I reset my password?",
        "What are the system requirements for endpoint security?",
        "How to configure firewall rules?",
        "Why is my SIEM dashboard not loading?",
        "How do I update my license?",
        "What ports need to be open for the agent?"
    ]
    
    print("\n🔍 Testing Help Questions:")
    print("-" * 40)
    
    for question in help_questions:
        print(f"\n❓ Question: {question}")
        
        response_data = await chatbot.get_help_response(question, max_context_docs=2)
        
        if "error" in response_data:
            print(f"   ❌ Error: {response_data['error']}")
            continue
        
        sources = response_data.get('sources', [])
        confidence = response_data.get('confidence', 'unknown')
        
        print(f"   🎯 Confidence: {confidence}")
        print(f"   📄 Sources found: {len(sources)}")
        
        for i, source in enumerate(sources[:2], 1):
            print(f"      [{i}] {source['title']}")
            print(f"          Similarity: {source['similarity']:.3f}")
            print(f"          Snippet: {source['snippet']}")
    
    print(f"\n✅ Help Chatbot Demo completed!")

# Example integration with LLM for complete responses
async def generate_complete_response(chatbot: ConfluenceHelpChatbot, question: str) -> str:
    """Generate a complete response using LLM + RAG context"""
    
    response_data = await chatbot.get_help_response(question)
    
    if "error" in response_data:
        return f"I encountered an error: {response_data['error']}"
    
    context = response_data.get('context', '')
    sources = response_data.get('sources', [])
    
    # In a real implementation, you'd send this to an LLM like Gemini
    llm_prompt = f"""
    You are a helpful technical support assistant for Trellix security products.
    
    User Question: {question}
    
    Relevant Documentation Context:
    {context}
    
    Instructions:
    - Provide a clear, helpful answer based on the context provided
    - If the context doesn't fully answer the question, acknowledge the limitation
    - Include specific steps or instructions when applicable
    - Be concise but comprehensive
    - Reference the source documentation when appropriate
    
    Response:
    """
    
    # Placeholder response (integrate with Gemini API in production)
    mock_response = f"""
    Based on our documentation, here's how to help with '{question}':
    
    {context[:500]}...
    
    Sources:
    """ + "\n".join([f"- {source['title']}" for source in sources[:3]])
    
    return mock_response

if __name__ == "__main__":
    asyncio.run(demo_help_chatbot())