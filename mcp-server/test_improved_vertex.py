#!/usr/bin/env python3
"""
Test script for improved Vertex AI service
Tests the enhanced prompt generation and action step parsing
"""

import asyncio
import json
from vertex_ai_service import VertexAIService

async def test_vertex_ai_improvements():
    """Test the improved Vertex AI service"""
    
    print("🧪 Testing Improved Vertex AI Service")
    print("=" * 50)
    
    # Initialize the service
    service = VertexAIService()
    await service.initialize()
    
    # Test cases with different types of requests
    test_cases = [
        {
            "name": "Simple Navigation",
            "request": "Go to available integrations page",
            "current_page": "your-integrations",
            "context": {
                "current_url": "http://localhost:3000/your-integrations",
                "page_title": "Integration Hub - Your Integrations",
                "page_elements": [
                    {
                        "tag": "button",
                        "text": "Add Integration",
                        "id": "add-integration-btn",
                        "testId": "add-integration",
                        "type": "button"
                    },
                    {
                        "tag": "button",
                        "text": "Available Integrations",
                        "className": "nav-button",
                        "testId": "nav-available"
                    }
                ]
            }
        },
        {
            "name": "Search Operation",
            "request": "Search for Slack integration",
            "current_page": "available-integrations",
            "context": {
                "current_url": "http://localhost:3000/available-integrations",
                "page_title": "Available Integrations",
                "page_elements": [
                    {
                        "tag": "input",
                        "placeholder": "Search integrations...",
                        "name": "search",
                        "type": "search"
                    },
                    {
                        "tag": "button",
                        "text": "Search",
                        "testId": "search-btn"
                    }
                ]
            }
        },
        {
            "name": "Complex Workflow",
            "request": "Add a new Slack integration and configure it",
            "current_page": "your-integrations",
            "context": {
                "current_url": "http://localhost:3000/your-integrations",
                "page_title": "Integration Hub",
                "page_elements": [
                    {
                        "tag": "button",
                        "text": "Add Integration",
                        "testId": "add-integration-btn"
                    },
                    {
                        "tag": "input",
                        "placeholder": "Search integrations...",
                        "type": "search"
                    }
                ]
            }
        }
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print("-" * 30)
        print(f"Request: {test_case['request']}")
        print(f"Current Page: {test_case['current_page']}")
        
        try:
            # Process the request
            response = await service.process_user_request(
                test_case['request'],
                test_case['context'],
                test_case['current_page']
            )
            
            # Display results
            print(f"✅ Success: {response.success}")
            print(f"📝 Message: {response.message}")
            print(f"🎯 Confidence: {response.confidence_score:.2f}")
            print(f"⏱️ Total Time: {response.estimated_total_time}ms")
            
            if response.warnings:
                print(f"⚠️ Warnings: {', '.join(response.warnings)}")
            
            print(f"🔧 Action Steps ({len(response.action_steps)}):")
            for j, step in enumerate(response.action_steps, 1):
                print(f"  {j}. {step.action_type.value}: {step.description}")
                print(f"     Target: {step.target}")
                if step.value:
                    print(f"     Value: {step.value}")
                print(f"     Priority: {step.priority.value}, Duration: {step.estimated_duration}ms")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    # Test validation functionality
    print("🔍 Testing Action Step Validation")
    print("-" * 30)
    
    if test_cases and len(test_cases) > 0:
        # Get action steps from first test case
        first_response = await service.process_user_request(
            test_cases[0]['request'],
            test_cases[0]['context'],
            test_cases[0]['current_page']
        )
        
        if first_response.action_steps:
            validation_result = await service.validate_action_steps(first_response.action_steps)
            print(f"Validation Result: {json.dumps(validation_result, indent=2)}")
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    asyncio.run(test_vertex_ai_improvements())
