#!/usr/bin/env python3
"""
Test RAG Implementation
Testing the new RAG capabilities in VertexAIService
"""

import asyncio
import os
from vertex_ai_service import VertexAIService

async def test_rag_functionality():
    """Test the RAG functionality"""
    
    print("🧪 Testing RAG Implementation")
    print("=" * 50)
    
    # Initialize the service
    service = VertexAIService()
    
    try:
        # Initialize the service
        print("1. Initializing Vertex AI Service with RAG...")
        await service.initialize()
        
        # Test health check
        print("2. Checking service health...")
        health = await service.health_check()
        print(f"   Health status: {health}")
        
        if health != "healthy":
            print("❌ Service is not healthy, cannot proceed with RAG tests")
            return
        
        # Test RAG context retrieval
        print("3. Testing RAG context retrieval...")
        test_queries = [
            "What is Trellix endpoint security?",
            "How does SIEM work?", 
            "What are network security features?",
            "Tell me about threat detection capabilities"
        ]
        
        for query in test_queries:
            print(f"\n📋 Testing query: '{query}'")
            
            # Test context retrieval
            context = await service.get_rag_context(query, n_results=3)
            if context:
                print(f"   ✅ Retrieved {len(context)} characters of context")
                print(f"   Preview: {context[:200]}...")
            else:
                print("   ⚠️ No context retrieved")
            
            # Test full RAG answer
            answer = await service.answer_question_with_rag(query)
            print(f"   📝 RAG Answer: {answer[:300]}...")
            print()
        
        # Test vector similarity search
        print("4. Testing vector similarity search...")
        similar_docs = await service.search_similar_documents("endpoint security protection", n_results=3)
        if similar_docs:
            print(f"   ✅ Found {len(similar_docs)} similar documents")
            for i, doc in enumerate(similar_docs, 1):
                print(f"   Doc {i}: Similarity={doc.get('similarity_score', 0):.3f}, Content preview: {doc.get('content', '')[:100]}...")
        else:
            print("   ⚠️ No similar documents found")
        
        print("\n✅ RAG testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during RAG testing: {e}")
        import traceback
        traceback.print_exc()

async def test_automation_with_rag():
    """Test automation request processing with RAG context"""
    
    print("\n🔧 Testing Automation with RAG Context")
    print("=" * 50)
    
    service = VertexAIService()
    
    try:
        await service.initialize()
        
        # Test automation request with RAG context
        test_request = "Help me find and configure endpoint security settings"
        
        print(f"🤖 Processing automation request: '{test_request}'")
        
        # Mock context for automation
        mock_context = {
            'page_title': 'Integration Management',
            'page_description': 'Manage your security integrations',
            'page_elements': [
                {'tag': 'button', 'text': 'Add Integration', 'testId': 'add-integration'},
                {'tag': 'input', 'type': 'search', 'testId': 'search-input'},
                {'tag': 'button', 'text': 'Search', 'testId': 'search-button'}
            ],
            'available_actions': ['search', 'add', 'configure']
        }
        
        response = await service.process_user_request(
            test_request, 
            context=mock_context,
            current_page="your-integrations"
        )
        
        print(f"✅ Automation response generated:")
        print(f"   Success: {response.success}")
        print(f"   Message: {response.message}")
        print(f"   Action Steps: {len(response.action_steps)}")
        
        for i, step in enumerate(response.action_steps, 1):
            print(f"   Step {i}: {step.action_type.value} - {step.description}")
        
    except Exception as e:
        print(f"❌ Error testing automation with RAG: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function"""
    print("🚀 Starting RAG Implementation Tests")
    print("=" * 60)
    
    # Test basic RAG functionality
    await test_rag_functionality()
    
    # Test automation with RAG
    await test_automation_with_rag()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())