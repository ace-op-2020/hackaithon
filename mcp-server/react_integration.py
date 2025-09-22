"""
React Integration Service
Service class to integrate MCP server with React chatbot
"""

class MCPClient:
    """Client for communicating with the MCP server from React"""
    
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url
    
    async def send_request(self, user_request, context=None, current_page=None):
        """Send a request to the MCP server"""
        
        # This is the JavaScript/React equivalent
        js_code = f"""
// MCP Client for React App
class MCPClient {{
    constructor(serverUrl = '{self.server_url}') {{
        this.serverUrl = serverUrl;
    }}
    
    async sendRequest(userRequest, context = {{}}, currentPage = null) {{
        try {{
            const payload = {{
                user_request: userRequest,
                context: {{
                    ...context,
                    current_url: window.location.href,
                    page_title: document.title,
                    timestamp: new Date().toISOString()
                }},
                current_page: currentPage || window.location.pathname,
                session_id: this.getSessionId()
            }};
            
            const response = await fetch(`${{this.serverUrl}}/mcp/request`, {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify(payload)
            }});
            
            if (!response.ok) {{
                throw new Error(`HTTP error! status: ${{response.status}}`);
            }}
            
            const result = await response.json();
            return result;
            
        }} catch (error) {{
            console.error('MCP request failed:', error);
            return {{
                success: false,
                message: `Failed to process request: ${{error.message}}`,
                action_steps: []
            }};
        }}
    }}
    
    async validateSteps(actionSteps) {{
        try {{
            const response = await fetch(`${{this.serverUrl}}/mcp/actions/validate`, {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify(actionSteps)
            }});
            
            return await response.json();
        }} catch (error) {{
            console.error('Validation failed:', error);
            return {{ valid: false, feedback: 'Validation request failed' }};
        }}
    }}
    
    async checkHealth() {{
        try {{
            const response = await fetch(`${{this.serverUrl}}/health`);
            return await response.json();
        }} catch (error) {{
            return {{ status: 'error', message: error.message }};
        }}
    }}
    
    getSessionId() {{
        // Generate or retrieve session ID
        let sessionId = localStorage.getItem('mcp_session_id');
        if (!sessionId) {{
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('mcp_session_id', sessionId);
        }}
        return sessionId;
    }}
}}

// Usage example in React component:
/*
import React, {{ useState }} from 'react';

const ChatbotComponent = () => {{
    const [mcpClient] = useState(() => new MCPClient());
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    
    const handleUserMessage = async (userMessage) => {{
        setLoading(true);
        
        try {{
            // Send request to MCP server
            const mcpResponse = await mcpClient.sendRequest(
                userMessage,
                {{
                    // Add any relevant context
                    user_preferences: getUserPreferences(),
                    page_data: getPageData()
                }}
            );
            
            if (mcpResponse.success) {{
                // Display the action steps to user
                const stepDescriptions = mcpResponse.action_steps.map(
                    step => `${{step.action_type}}: ${{step.description}}`
                ).join('\\n');
                
                setMessages(prev => [...prev, {{
                    type: 'bot',
                    content: `I can help you with that! Here's what I'll do:\\n\\n${{stepDescriptions}}`,
                    actionSteps: mcpResponse.action_steps
                }}]);
                
                // Optionally execute the steps
                if (window.confirm('Would you like me to execute these steps?')) {{
                    await executeActionSteps(mcpResponse.action_steps);
                }}
            }} else {{
                setMessages(prev => [...prev, {{
                    type: 'bot',
                    content: `Sorry, I couldn't process that request: ${{mcpResponse.message}}`
                }}]);
            }}
            
        }} catch (error) {{
            setMessages(prev => [...prev, {{
                type: 'bot',
                content: 'Sorry, there was an error processing your request.'
            }}]);
        }} finally {{
            setLoading(false);
        }}
    }};
    
    const executeActionSteps = async (actionSteps) => {{
        for (const step of actionSteps) {{
            try {{
                await executeStep(step);
                await new Promise(resolve => setTimeout(resolve, step.estimated_duration || 1000));
            }} catch (error) {{
                console.error(`Failed to execute step ${{step.id}}:`, error);
                break;
            }}
        }}
    }};
    
    const executeStep = async (step) => {{
        switch (step.action_type) {{
            case 'click':
                const clickElement = document.querySelector(step.target);
                if (clickElement) clickElement.click();
                break;
                
            case 'type':
                const typeElement = document.querySelector(step.target);
                if (typeElement) {{
                    typeElement.value = step.value;
                    typeElement.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
                break;
                
            case 'navigate':
                window.location.href = step.target;
                break;
                
            case 'scroll':
                const scrollElement = document.querySelector(step.target);
                if (scrollElement) scrollElement.scrollIntoView({{ behavior: 'smooth' }});
                break;
                
            case 'wait':
                await new Promise(resolve => setTimeout(resolve, step.estimated_duration || 1000));
                break;
                
            case 'validate':
                const validateElement = document.querySelector(step.target);
                if (!validateElement) throw new Error(`Element not found: ${{step.target}}`);
                break;
                
            default:
                console.warn(`Unknown action type: ${{step.action_type}}`);
        }}
    }};
    
    return (
        <div className="chatbot-container">
            {{/* Your chatbot UI here */}}
        </div>
    );
}};

export default ChatbotComponent;
*/

// Export the MCP client for use in your React app
export default MCPClient;
"""
        
        return js_code

# Example usage
if __name__ == "__main__":
    client = MCPClient()
    js_integration = client.send_request("example", {{}}, "/")
    print("JavaScript integration code generated!")
    print("Copy the JavaScript code above into your React project.")
