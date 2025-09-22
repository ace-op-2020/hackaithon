# MCP Server

A lightweight Model Context Protocol (MCP) server built with FastAPI and integrated with Google Cloud Vertex AI. This server processes natural language requests from a React chatbot and returns structured action steps for web automation.

## Features

- **FastAPI**: Fast, modern web framework for building APIs
- **Vertex AI Integration**: Uses Google Cloud's Gemini models for intelligent request processing
- **MCP Protocol**: Implements Model Context Protocol for structured communication
- **CORS Support**: Configured for React frontend integration
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
