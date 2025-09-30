"""
RAG Integration Example
Shows how to integrate the new RAG capabilities into existing applications
"""

from vertex_ai_service import VertexAIService
from typing import Dict, Any, Optional
import asyncio

class TrellixHelpChatbot:
    """
    Enhanced chatbot using RAG for Trellix help and documentation queries
    """
    
    def __init__(self):
        self.vertex_service = VertexAIService()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the RAG-enabled chatbot"""
        try:
            await self.vertex_service.initialize()
            self.initialized = True
            print("✅ Trellix Help Chatbot with RAG initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {e}")
            return False
    
    async def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with RAG context
        
        Args:
            question: The user's question
            
        Returns:
            Dict with answer, sources, and confidence information
        """
        if not self.initialized:
            raise Exception("Chatbot not initialized")
        
        try:
            # Get RAG-enhanced answer
            answer = await self.vertex_service.answer_question_with_rag(question)
            
            # Get relevant documents for source attribution
            similar_docs = await self.vertex_service.search_similar_documents(question, n_results=3)
            
            # Extract source information
            sources = []
            for doc in similar_docs:
                metadata = doc.get('metadata', {})
                sources.append({
                    'title': metadata.get('title', 'Unknown'),
                    'source': metadata.get('source', 'Unknown'),
                    'similarity': doc.get('similarity_score', 0),
                    'preview': doc.get('content', '')[:150] + '...'
                })
            
            # Calculate confidence based on similarity scores
            confidence = 'high' if similar_docs and similar_docs[0].get('similarity_score', 0) > 0.7 else 'medium' if similar_docs else 'low'
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'has_knowledge_base_context': len(similar_docs) > 0
            }
            
        except Exception as e:
            return {
                'answer': f"I apologize, but I encountered an error: {str(e)}",
                'sources': [],
                'confidence': 'low',
                'has_knowledge_base_context': False
            }
    
    async def get_automation_help(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get help with automation using RAG context
        
        Args:
            user_request: What the user wants to automate
            context: Current application context
            
        Returns:
            Automation response with enhanced context
        """
        if not self.initialized:
            raise Exception("Chatbot not initialized")
        
        try:
            # Process automation request with RAG context
            response = await self.vertex_service.process_user_request(
                user_request, 
                context=context
            )
            
            # Get additional help information using RAG
            help_context = await self.vertex_service.get_rag_context(user_request, n_results=2)
            
            return {
                'automation_response': response,
                'help_context': help_context,
                'success': response.success
            }
            
        except Exception as e:
            return {
                'automation_response': None,
                'help_context': '',
                'success': False,
                'error': str(e)
            }

# Example usage functions
async def demo_help_chatbot():
    """Demonstrate the RAG-enabled help chatbot"""
    print("🤖 Trellix Help Chatbot Demo with RAG")
    print("=" * 50)
    
    chatbot = TrellixHelpChatbot()
    
    # Initialize chatbot
    if not await chatbot.initialize():
        print("❌ Failed to initialize chatbot")
        return
    
    # Example questions
    questions = [
        "What is Trellix endpoint security and how does it work?",
        "How do I configure SIEM alert policies?",
        "What are the best practices for network security setup?",
        "How do I troubleshoot endpoint agent connectivity issues?",
        "What integration options are available for third-party tools?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        
        result = await chatbot.ask_question(question)
        
        print(f"🎯 Confidence: {result['confidence']}")
        print(f"📚 Knowledge Base Context: {'Yes' if result['has_knowledge_base_context'] else 'No'}")
        print(f"💬 Answer: {result['answer'][:300]}...")
        
        if result['sources']:
            print("📖 Sources:")
            for i, source in enumerate(result['sources'][:2], 1):
                print(f"   {i}. {source['title']} (Similarity: {source['similarity']:.3f})")
                print(f"      Preview: {source['preview']}")
        print("-" * 50)

async def demo_automation_with_rag():
    """Demonstrate automation with RAG context"""
    print("\n🔧 Automation with RAG Context Demo")
    print("=" * 50)
    
    chatbot = TrellixHelpChatbot()
    
    if not await chatbot.initialize():
        print("❌ Failed to initialize chatbot")
        return
    
    # Mock application context
    mock_context = {
        'page_title': 'Security Dashboard',
        'page_description': 'Trellix Security Management Console',
        'page_elements': [
            {'tag': 'button', 'text': 'Configure Endpoint', 'testId': 'config-endpoint'},
            {'tag': 'button', 'text': 'View Alerts', 'testId': 'view-alerts'},
            {'tag': 'input', 'placeholder': 'Search policies...', 'testId': 'policy-search'}
        ],
        'available_actions': ['configure', 'monitor', 'search']
    }
    
    automation_requests = [
        "Help me configure endpoint security policies",
        "I need to set up SIEM alerting rules",
        "Show me how to add a new security integration"
    ]
    
    for request in automation_requests:
        print(f"\n🤖 Automation Request: {request}")
        
        result = await chatbot.get_automation_help(request, mock_context)
        
        if result['success']:
            response = result['automation_response']
            print(f"✅ Success: {response.message}")
            print(f"📋 Generated {len(response.action_steps)} action steps")
            
            if result['help_context']:
                print(f"📚 Additional Context Available: {len(result['help_context'])} characters")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        print("-" * 50)

async def main():
    """Main demo function"""
    print("🚀 RAG Integration Demonstration")
    print("=" * 60)
    
    # Demo help chatbot with RAG
    await demo_help_chatbot()
    
    # Demo automation with RAG
    await demo_automation_with_rag()
    
    print("\n🎉 RAG Integration Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())