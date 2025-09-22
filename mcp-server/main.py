"""
Lightweight MCP Server using FastAPI and Vertex AI
Handles user requests from React chatbot and returns structured action steps
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from vertex_ai_service import VertexAIService
from mcp_protocol import MCPRequest, MCPResponse, ActionStep

# Initialize FastAPI app
app = FastAPI(
    title="MCP Server",
    description="Model Context Protocol Server for React App Automation",
    version="1.0.0"
)

# Configure CORS for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Vertex AI service
vertex_service = VertexAIService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 MCP Server starting up...")
    await vertex_service.initialize()
    print("✅ MCP Server ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MCP Server is running",
        "timestamp": datetime.now().isoformat(),
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "vertex_ai": await vertex_service.health_check(),
            "mcp_server": "running"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/mcp/request", response_model=MCPResponse)
async def handle_mcp_request(request: MCPRequest):
    """
    Main MCP endpoint to handle user requests and return action steps
    """
    try:
        print(f"📨 Received MCP request: {request.user_request}")
        
        # Process the request through Vertex AI
        response = await vertex_service.process_user_request(
            user_request=request.user_request,
            context=request.context,
            current_page=request.current_page
        )
        
        return response
        
    except Exception as e:
        print(f"❌ Error processing MCP request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/actions/validate")
async def validate_action_steps(steps: List[ActionStep]):
    """
    Validate if the proposed action steps are feasible
    """
    try:
        validation_result = await vertex_service.validate_action_steps(steps)
        return {
            "valid": validation_result["valid"],
            "feedback": validation_result["feedback"],
            "suggestions": validation_result.get("suggestions", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/execute")
async def execute_action_plan(request: dict):
    """
    Execute a validated action plan (placeholder for future implementation)
    """
    return {
        "status": "execution_planned",
        "message": "Action execution is not yet implemented",
        "steps_count": len(request.get("steps", []))
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
