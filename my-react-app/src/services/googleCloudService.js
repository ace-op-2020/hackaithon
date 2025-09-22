// Browser-compatible Google Cloud Vertex AI service
// Using REST API directly instead of Node.js client libraries

class GoogleCloudService {
  constructor() {
    this.projectId = process.env.REACT_APP_GOOGLE_CLOUD_PROJECT_ID || 'svc-hackathon-prod15';
    this.location = process.env.REACT_APP_GOOGLE_CLOUD_REGION || 'us-central1';
    this.model = process.env.REACT_APP_GEMINI_MODEL || 'gemini-1.5-flash';
    this.accessToken = process.env.REACT_APP_GOOGLE_CLOUD_ACCESS_TOKEN;
    this.apiKey = process.env.REACT_APP_GOOGLE_API_KEY;
    this.serviceAccountPath = process.env.REACT_APP_GOOGLE_CLOUD_SERVICE_ACCOUNT_PATH;
    this.initialized = false;
    this.cachedAccessToken = null;
    this.tokenExpiry = null;
  }

  async getServiceAccountToken() {
    // Check if we have a cached token that's still valid
    if (this.cachedAccessToken && this.tokenExpiry && Date.now() < this.tokenExpiry) {
      console.log('🔄 [Auth] Using cached access token');
      return this.cachedAccessToken;
    }

    try {
      console.log('🔑 [Auth] Getting new access token from service account...');
      
      // For browser environments, we can't use service account files directly
      // Instead, we'll use the API key or pre-generated access token
      if (this.accessToken) {
        console.log('✅ [Auth] Using provided access token');
        this.cachedAccessToken = this.accessToken;
        // Tokens typically expire in 1 hour, cache for 50 minutes to be safe
        this.tokenExpiry = Date.now() + (50 * 60 * 1000);
        return this.cachedAccessToken;
      }
      
      // If we have an API key, we can use that instead
      if (this.apiKey) {
        console.log('✅ [Auth] Using API key for authentication');
        return this.apiKey;
      }
      
      return null;
    } catch (error) {
      console.warn('⚠️ [Auth] Failed to get service account token:', error.message);
      return null;
    }
  }

  async initialize() {
    if (this.initialized) return;

    try {
      // Try to get service account token first
      const serviceAccountToken = await this.getServiceAccountToken();
      if (serviceAccountToken) {
        this.accessToken = serviceAccountToken;
        console.log('✅ [Gemini Setup] Using service account authentication');
      } else if (!this.accessToken && (!this.apiKey || this.apiKey === 'test_key_for_development' || this.apiKey === 'your_api_key_here')) {
        console.warn('⚠️ [Gemini Setup] No valid Google Cloud credentials found.');
        console.warn('📋 [Gemini Setup] To enable Gemini AI integration:');
        console.warn('   1. Use service account: Place service account JSON file in project root');
        console.warn('   2. OR get an API key from https://makersuite.google.com/app/apikey');
        console.warn('   3. Update .env with credentials and restart server (npm start)');
        console.warn('🔄 [Gemini Setup] For now, using pattern matching parser (this still works!)');
      }

      this.initialized = true;
      console.log('Google Cloud service initialized for browser environment');
    } catch (error) {
      console.error('Failed to initialize Google Cloud service:', error);
      throw error;
    }
  }

  async callGemini(prompt, systemPrompt = null) {
    try {
      console.log('🚀 [Gemini] Starting API call...');
      console.log('📝 [Gemini] Prompt:', prompt);
      console.log('🔧 [Gemini] System Prompt:', systemPrompt);
      
      if (!this.initialized) {
        console.log('⚙️ [Gemini] Initializing service...');
        await this.initialize();
      }

      // Prepare the request content
      const contents = [];
      
      if (systemPrompt) {
        contents.push({
          role: 'user',
          parts: [{ text: systemPrompt }]
        });
      }
      
      contents.push({
        role: 'user',
        parts: [{ text: prompt }]
      });

      const requestBody = {
        contents: contents,
        generationConfig: {
          temperature: 0.1,
          topK: 40,
          topP: 0.95,
          maxOutputTokens: 1024,
        },
        safetySettings: [
          {
            category: "HARM_CATEGORY_HARASSMENT",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          },
          {
            category: "HARM_CATEGORY_HATE_SPEECH",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          },
          {
            category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          },
          {
            category: "HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          }
        ]
      };

      // Try with access token first, then fallback to API key
      let response;
      const headers = { 'Content-Type': 'application/json' };
      
      console.log('🔑 [Gemini] Auth method check - Access Token:', !!this.accessToken, 'API Key:', !!this.apiKey);
      console.log('📡 [Gemini] Request body:', JSON.stringify(requestBody, null, 2));
      
      if (this.accessToken) {
        console.log('🔐 [Gemini] Using access token authentication');
        headers['Authorization'] = `Bearer ${this.accessToken}`;
        const url = `https://${this.location}-aiplatform.googleapis.com/v1/projects/${this.projectId}/locations/${this.location}/publishers/google/models/${this.model}:generateContent`;
        console.log('🌐 [Gemini] API URL:', url);
        
        response = await fetch(url, {
          method: 'POST',
          headers: headers,
          body: JSON.stringify(requestBody)
        });
      } else if (this.apiKey) {
        // Check if it's a test/placeholder key
        if (this.apiKey === 'test_key_for_development' || this.apiKey === 'your_api_key_here') {
          console.log('🔄 [Gemini] Test API key detected, skipping API call and using fallback parser');
          throw new Error('Test API key - using fallback parser');
        }
        
        console.log('🔑 [Gemini] Using API key authentication');
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${this.model}:generateContent?key=${this.apiKey}`;
        console.log('🌐 [Gemini] API URL:', url);
        
        response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody)
        });
      } else {
        const error = 'No valid authentication method available. Please set REACT_APP_GOOGLE_CLOUD_ACCESS_TOKEN or REACT_APP_GOOGLE_API_KEY';
        console.log('🔄 [Gemini] No valid auth, will use fallback parser');
        throw new Error(error);
      }

      console.log('📊 [Gemini] Response status:', response.status, response.statusText);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ [Gemini] API error response:', errorText);
        throw new Error(`Google AI API error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log('📨 [Gemini] Full response data:', JSON.stringify(data, null, 2));
      
      if (data.candidates && data.candidates[0] && data.candidates[0].content) {
        const result = data.candidates[0].content.parts[0].text;
        console.log('✅ [Gemini] Extracted text result:', result);
        return result;
      } else {
        console.error('❌ [Gemini] Invalid response structure:', data);
        throw new Error('Invalid response format from Google AI API');
      }

    } catch (error) {
      console.error('❌ [Gemini] API call failed:', error);
      console.error('❌ [Gemini] Error details:', error.message);
      console.error('❌ [Gemini] Stack trace:', error.stack);
      throw error;
    }
  }

  // Specialized method for parsing browser automation commands
  async parseBrowserCommand(userCommand) {
    console.log('🎯 [BrowserCommand] Starting to parse command:', userCommand);
    
    const systemPrompt = `You are a browser automation assistant that can handle both browser actions and app navigation. Parse user commands and return ONLY valid JSON with this exact structure:
    {
      "action": "navigate|click|fill|screenshot|search|select|app-navigate|unknown",
      "params": {
        "url": "string for navigate",
        "selector": "string for click/fill/screenshot",
        "value": "string for fill",
        "query": "string for search",
        "option": "string for select",
        "page": "string for app-navigate (llm-demo|browser-automation|available-integrations|your-integrations|gemini-test)"
      },
      "confidence": "high|medium|low",
      "description": "human-readable description of what will happen"
    }

    Rules:
    - For browser navigation: extract URL from commands like "go to", "navigate to", "open", "visit"
    - For app navigation: use "app-navigate" action for internal navigation like:
      * "navigate to llm demo" -> page: "llm-demo"
      * "go to browser automation" -> page: "browser-automation"  
      * "show available integrations" -> page: "available-integrations"
      * "go to your integrations" or "go home" -> page: "your-integrations"
      * "open gemini test" or "test gemini" -> page: "gemini-test"
    - For clicking: extract element description from "click on", "tap", "press"
    - For filling: extract field and value from "fill X with Y", "type Y in X", "enter Y in X"
    - For screenshots: extract target from "screenshot of X", "capture X"
    - For search: extract query from "search for X", "find X"
    - For select: extract option from "select X", "choose X"
    - Return "unknown" action if command is unclear
    - Be specific with selectors when possible
    - Confidence should reflect how certain you are about the action`;

    try {
      console.log('📞 [BrowserCommand] Calling Gemini API...');
      const response = await this.callGemini(userCommand, systemPrompt);
      console.log('📝 [BrowserCommand] Raw Gemini response:', response);
      
      // Try to parse the JSON response
      try {
        console.log('🔍 [BrowserCommand] Attempting to parse JSON...');
        const parsed = JSON.parse(response);
        console.log('✅ [BrowserCommand] Successfully parsed JSON:', JSON.stringify(parsed, null, 2));
        
        // Validate the response structure
        if (parsed.action && parsed.params && parsed.confidence && parsed.description) {
          console.log('✅ [BrowserCommand] Response structure validated');
          return parsed;
        } else {
          console.error('❌ [BrowserCommand] Invalid response structure - missing required fields');
          throw new Error('Invalid response structure');
        }
      } catch (parseError) {
        console.error('❌ [BrowserCommand] Failed to parse Gemini response as JSON:', parseError);
        console.log('🔄 [BrowserCommand] Raw response that failed to parse:', response);
        console.log('🔄 [BrowserCommand] Falling back to pattern matching...');
        
        // Fallback to pattern matching
        return this.fallbackCommandParser(userCommand);
      }
      
    } catch (error) {
      console.error('❌ [BrowserCommand] Gemini API call failed:', error);
      console.log('🔄 [BrowserCommand] Using fallback parser...');
      return this.fallbackCommandParser(userCommand);
    }
  }

  // Fallback command parser for when Gemini is unavailable
  fallbackCommandParser(command) {
    console.log('🔧 [FallbackParser] Using fallback parser for command:', command);
    const lowerCommand = command.toLowerCase();
    
    // Define common patterns and their corresponding actions
    const patterns = [
      // App navigation patterns (internal navigation)
      {
        pattern: /(?:navigate to|go to|open|show)\s+(?:the\s+)?(?:llm|ai|gemini|command)\s+(?:demo|page|component)/i,
        action: 'app-navigate',
        extract: () => ({ page: 'llm-demo' })
      },
      {
        pattern: /(?:navigate to|go to|open|show)\s+(?:the\s+)?(?:browser|automation)\s+(?:demo|page|component)/i,
        action: 'app-navigate',
        extract: () => ({ page: 'browser-automation' })
      },
      {
        pattern: /(?:navigate to|go to|open|show)\s+(?:the\s+)?(?:available|integrations?)\s+(?:page|component)?/i,
        action: 'app-navigate',
        extract: () => ({ page: 'available-integrations' })
      },
      {
        pattern: /(?:navigate to|go to|open|show)\s+(?:the\s+)?(?:your|my)\s+(?:integrations?)\s+(?:page|component)?/i,
        action: 'app-navigate',
        extract: () => ({ page: 'your-integrations' })
      },
      {
        pattern: /(?:go\s+)?(?:home|back)/i,
        action: 'app-navigate',
        extract: () => ({ page: 'your-integrations' })
      },
      {
        pattern: /(?:open|show|test)\s+(?:the\s+)?(?:gemini|test)\s+(?:test|component|page)?/i,
        action: 'app-navigate',
        extract: () => ({ page: 'gemini-test' })
      },
      // Browser navigation patterns (external URLs)
      {
        pattern: /(?:go to|navigate to|open|visit)\s+(.+)/i,
        action: 'navigate',
        extract: (match) => ({ url: match[1] })
      },
      {
        pattern: /(?:click|tap|press)\s+(?:on\s+)?(.+)/i,
        action: 'click',
        extract: (match) => ({ selector: match[1] })
      },
      {
        pattern: /(?:fill|type|enter)\s+(.+?)\s+(?:with|as|in)\s+(.+)/i,
        action: 'fill',
        extract: (match) => ({ selector: match[1], value: match[2] })
      },
      {
        pattern: /(?:take|capture)\s+(?:a\s+)?screenshot(?:\s+of\s+(.+))?/i,
        action: 'screenshot',
        extract: (match) => ({ selector: match[1] || null })
      },
      {
        pattern: /(?:search|find)\s+(?:for\s+)?(.+)/i,
        action: 'search',
        extract: (match) => ({ query: match[1] })
      },
      {
        pattern: /(?:select|choose)\s+(.+)/i,
        action: 'select',
        extract: (match) => ({ option: match[1] })
      }
    ];

    // Try to match patterns
    console.log('🔍 [FallbackParser] Testing patterns against command...');
    for (const { pattern, action, extract } of patterns) {
      const match = command.match(pattern);
      console.log(`🔍 [FallbackParser] Pattern ${pattern} -> Match:`, !!match);
      if (match) {
        const result = {
          action,
          params: extract(match),
          confidence: 'high',
          description: `Execute ${action} action`
        };
        console.log('✅ [FallbackParser] Pattern matched! Result:', result);
        return result;
      }
    }

    console.log('🔍 [FallbackParser] No patterns matched, checking fallback intents...');
    
    // Fallback: try to understand intent
    if (lowerCommand.includes('aws') || lowerCommand.includes('integration')) {
      const result = {
        action: 'navigate',
        params: { url: 'http://localhost:3000' },
        confidence: 'medium',
        description: 'Navigate to TrelliX app'
      };
      console.log('✅ [FallbackParser] Fallback intent matched:', result);
      return result;
    }

    const unknownResult = {
      action: 'unknown',
      params: {},
      confidence: 'low',
      description: 'Command not understood'
    };
    console.log('❌ [FallbackParser] No match found, returning unknown:', unknownResult);
    return unknownResult;
  }
}

// Create and export a singleton instance
const googleCloudService = new GoogleCloudService();
export default googleCloudService; 