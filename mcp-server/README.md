# Trellix Knowledge Graph Builder

A comprehensive Python framework for building and maintaining a knowledge graph of Trellix's product offerings using Confluence PRDs, website data, and other documentation sources.

## 🌟 Features

- **Multi-Source Data Extraction**: Extract data from Confluence, websites, and document repositories
- **AI-Powered Graph Construction**: Use Gemini API and LangChain's LLMGraphTransformer for intelligent entity and relationship extraction
- **Scalable Storage**: Store knowledge graphs and embeddings in Google Cloud Spanner Graph database
- **Advanced Text Processing**: Segment content using RecursiveCharacterTextSplitter and Document AI Layout Parser
- **Vector Embeddings**: Generate and store vector embeddings using Vertex AI Embeddings APIs
- **Graph Analytics**: Comprehensive graph analysis, visualization, and query capabilities
- **Configurable Pipeline**: Flexible configuration system supporting environment variables and YAML/JSON files
- **MCP Server Integration**: Implements Model Context Protocol for chatbot integration

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Data Sources      │    │   Processing        │    │   Storage           │
│                     │    │                     │    │                     │
│ • Confluence API    │───▶│ • Text Segmentation │───▶│ • Spanner Graph DB  │
│ • Website Scraping  │    │ • Entity Extraction │    │ • Vector Store      │
│ • Document Upload   │    │ • Relationship Det. │    │ • Metadata Store    │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                      │
                           ┌─────────────────────┐
                           │   AI Services       │
                           │                     │
                           │ • Vertex AI Gemini  │
                           │ • Embeddings API    │
                           │ • LangChain Tools   │
                           └─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Cloud Project with Vertex AI and Spanner APIs enabled
- Service account key with appropriate permissions
- (Optional) Confluence access credentials

### Installation

1. **Run the setup script**:
   ```bash
   python setup.py
   ```

2. **Activate the virtual environment**:
   - Windows: `venv\\Scripts\\activate`
   - Linux/Mac: `source venv/bin/activate`

3. **Configure the system**:
   ```bash
   # Copy example configuration files
   cp .env.example .env
   cp config.yaml.example config.yaml
   
   # Edit with your actual credentials and settings
   notepad .env          # Windows
   nano .env            # Linux/Mac
   ```

### Basic Usage

```python
import asyncio
from knowledge_grapth_builder import TrellixKnowledgeGraphBuilder
from config_manager import get_default_config

async def main():
    # Load configuration
    config = get_default_config()
    
    # Update with your specific settings
    config.confluence.spaces = ["PROD", "DOCS", "HELP"]
    config.web_scraping.domains = [
        "https://www.trellix.com",
        "https://docs.trellix.com"
    ]
    
    # Build knowledge graph
    builder = TrellixKnowledgeGraphBuilder(config)
    success = await builder.build_knowledge_graph()
    
    if success:
        print("✅ Knowledge graph built successfully!")
    else:
        print("❌ Failed to build knowledge graph")

if __name__ == "__main__":
    asyncio.run(main())
```

## ⚙️ Configuration

### Environment Variables

```bash
# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
VERTEX_AI_PROJECT_ID=your-project-id
VERTEX_AI_LOCATION=us-central1

# Confluence Configuration
CONFLUENCE_URL=https://your-company.atlassian.net/wiki
CONFLUENCE_USERNAME=your-email@company.com
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACES=PROD,DOCS,HELP

# Web Scraping Configuration
WEB_SCRAPING_DOMAINS=https://www.trellix.com,https://docs.trellix.com

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
LOG_LEVEL=INFO
```

## 📊 Knowledge Graph Components

### Core Framework Files

1. **`knowledge_grapth_builder.py`** - Main framework with complete pipeline
2. **`config_manager.py`** - Configuration management system
3. **`graph_utils.py`** - Graph analysis and visualization utilities
4. **`setup.py`** - Automated setup and dependency installation
5. **`test_knowledge_graph.py`** - Comprehensive test suite

### Database Schema

The system creates the following tables in Spanner:

- **Nodes**: Stores graph nodes with properties and embeddings
- **Relationships**: Stores graph edges with relationship types and confidence scores
- **Documents**: Stores original document metadata
- **TextChunks**: Stores text segments with vector embeddings

## 🧠 AI-Powered Processing

### Text Segmentation

Intelligent text chunking for optimal processing:

```python
from knowledge_grapth_builder import DocumentProcessor

processor = DocumentProcessor(config)
processed_docs = await processor.process_documents(raw_documents)
```

### Knowledge Graph Construction

Use Vertex AI Gemini for entity and relationship extraction:

```python
from knowledge_grapth_builder import VertexAIGraphBuilder

graph_builder = VertexAIGraphBuilder(config)
await graph_builder.initialize()

knowledge_graph = await graph_builder.build_graph(processed_documents)
embeddings = await graph_builder.generate_embeddings(text_chunks)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
python -m pytest test_knowledge_graph.py -v

# Run with coverage
python -m pytest test_knowledge_graph.py --cov=knowledge_grapth_builder
```

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

**2. Authentication Errors**
```bash
# Set credentials environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

**3. Spanner Connection Issues**
- Verify Spanner instance exists
- Check IAM permissions
- Ensure Spanner API is enabled

## 🔧 MCP Server Integration

The knowledge graph integrates with the existing MCP server for chatbot functionality:

```python
# The existing MCP server files work alongside the knowledge graph:
# - main.py (FastAPI server)
# - vertex_ai_service.py (AI processing)
# - mcp_protocol.py (Protocol definitions)
```

---

Built with ❤️ for the Trellix Security Platform
- **Action Planning**: Converts natural language requests into executable action steps
- **Validation**: Validates action plans for feasibility

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the `.env` file and update the credentials path:

```bash
# Make sure your Google Cloud credentials are properly set
GOOGLE_APPLICATION_CREDENTIALS=../python_test_app/credentials.json
```

### 3. Run the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or using the npm script
python main.py
```

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health status

### MCP Protocol
- `POST /mcp/request` - Process user requests and return action steps
- `POST /mcp/actions/validate` - Validate action step feasibility
- `POST /mcp/execute` - Execute action plans (placeholder)

## Request/Response Format

### MCP Request
```json
{
  "user_request": "Click on the login button and enter my credentials",
  "context": {
    "current_user": "john@example.com",
    "page_elements": ["login-form", "username-input"]
  },
  "current_page": "/login",
  "session_id": "session-123"
}
```

### MCP Response
```json
{
  "success": true,
  "message": "Generated 3 action steps for: Click on the login button and enter my credentials",
  "action_steps": [
    {
      "id": "step_1",
      "action_type": "click",
      "description": "Click on the login button",
      "target": "button[data-testid='login-btn']",
      "priority": "high",
      "estimated_duration": 500
    }
  ],
  "estimated_total_time": 2500,
  "confidence_score": 0.85
}
```

## Action Types

- `click` - Click on elements
- `type` - Type text into inputs
- `navigate` - Navigate to pages
- `scroll` - Scroll to elements
- `wait` - Wait for conditions
- `validate` - Validate element states
- `extract` - Extract data from elements
- `custom` - Custom JavaScript execution

## Integration with React App

The MCP server is designed to work with your React chatbot. Here's how to integrate:

```javascript
// Example React integration
const sendMCPRequest = async (userRequest) => {
  const response = await fetch('http://localhost:8000/mcp/request', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_request: userRequest,
      current_page: window.location.pathname,
      context: {
        // Add relevant context
      }
    })
  });
  
  const result = await response.json();
  return result.action_steps;
};
```

## Development

### Project Structure
```
mcp-server/
├── main.py              # FastAPI application
├── mcp_protocol.py      # MCP data models
├── vertex_ai_service.py # Vertex AI integration
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
└── README.md           # This file
```

### Adding New Action Types

1. Add the new action type to `ActionType` enum in `mcp_protocol.py`
2. Update the prompt in `vertex_ai_service.py` to include the new action
3. Test with your React app

## Troubleshooting

1. **Vertex AI Authentication**: Ensure your credentials file path is correct
2. **CORS Issues**: Check that your React app URL is in the allowed origins
3. **Model Errors**: Verify your Google Cloud project has Vertex AI API enabled

## License

MIT License
