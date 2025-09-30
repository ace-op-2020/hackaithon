"""
Vertex AI Service
Handles communication with Google Cloud Vertex AI for LLM processing
"""

import os
import json
from typing import Dict, Any, List, Optional
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
from google.cloud import spanner

from mcp_protocol import MCPResponse, ActionStep, ActionType, Priority

class VertexAIService:
    """Service for interacting with Vertex AI"""
    
    def __init__(self):
        self.project_id = "svc-hackathon-prod15"
        self.location = "us-central1"
        self.model_name = "gemini-2.0-flash-lite-001"
        self.embedding_model_name = "text-embedding-004"
        self.model = None
        self.embedding_model = None
        self.spanner_client = None
        self.spanner_instance_id = "trellix-knowledge-graph"
        self.spanner_database_id = "knowledge_graph_db"
        self.vector_table_name = "trellix_document_embeddings"
        self.initialized = False
        self.rag_enabled = True
    
    async def initialize(self):
        """Initialize the Vertex AI service with RAG capabilities"""
        try:
            # Set credentials path - using the file from root directory
            credentials_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "svc-hackathon-prod15-534bc641841c.json")
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
            print(f"🔐 Using credentials file: {credentials_path}")
            
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            
            # Load the generative model
            self.model = GenerativeModel(self.model_name)
            
            # Load the embedding model for RAG
            self.embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model_name)
            
            # Initialize Spanner client for vector database access
            self.spanner_client = spanner.Client(project=self.project_id)
            
            self.initialized = True
            
            print(f"✅ Vertex AI initialized with model: {self.model_name}")
            print(f"✅ Embedding model initialized: {self.embedding_model_name}")
            print(f"✅ RAG capabilities enabled with Spanner vector database")
            
        except Exception as e:
            print(f"❌ Failed to initialize Vertex AI with RAG: {e}")
            # Try to initialize without RAG as fallback
            try:
                vertexai.init(project=self.project_id, location=self.location)
                self.model = GenerativeModel(self.model_name)
                self.initialized = True
                self.rag_enabled = False
                print(f"⚠️ Vertex AI initialized without RAG capabilities")
            except Exception as fallback_error:
                print(f"❌ Failed to initialize Vertex AI even without RAG: {fallback_error}")
                raise fallback_error
    
    async def health_check(self) -> str:
        """Check if the service is healthy"""
        if not self.initialized:
            return "not_initialized"
        
        try:
            # Simple test query
            response = self.model.generate_content("Hello")
            return "healthy" if response else "error"
        except Exception:
            return "error"
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text using Vertex AI"""
        if not self.rag_enabled or not self.embedding_model:
            return []
        
        try:
            embeddings = []
            batch_size = 100
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.get_embeddings(batch)
                embeddings.extend([emb.values for emb in batch_embeddings])
            
            return embeddings
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return []
    
    async def search_similar_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in the Spanner vector database"""
        if not self.rag_enabled or not self.spanner_client:
            print("⚠️ RAG is disabled or Spanner client not available")
            return []
        
        try:
            # Generate embedding for the query
            query_embeddings = await self.generate_embeddings([query])
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            
            # Search in Spanner vector database
            instance = self.spanner_client.instance(self.spanner_instance_id)
            database = instance.database(self.spanner_database_id)
            
            # Query to find similar documents using cosine similarity
            # Note: This is a simplified approach - for production, use Spanner's vector search capabilities
            with database.snapshot() as snapshot:
                results = snapshot.execute_sql(f"""
                    SELECT 
                        chunk_id,
                        document_id, 
                        content,
                        metadata,
                        embedding_vector
                    FROM {self.vector_table_name}
                    LIMIT 100
                """)
                
                documents = []
                for row in results:
                    try:
                        stored_embedding = json.loads(row[4]) if row[4] else []
                        if stored_embedding:
                            # Calculate cosine similarity
                            similarity = self._cosine_similarity(query_embedding, stored_embedding)
                            
                            documents.append({
                                'chunk_id': row[0],
                                'document_id': row[1], 
                                'content': row[2],
                                'metadata': json.loads(row[3]) if row[3] else {},
                                'similarity_score': similarity
                            })
                    except Exception as e:
                        print(f"⚠️ Error processing document row: {e}")
                        continue
                
                # Sort by similarity and return top results
                documents.sort(key=lambda x: x['similarity_score'], reverse=True)
                return documents[:n_results]
                
        except Exception as e:
            print(f"❌ Error searching similar documents: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import math
            
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0.0
    
    async def get_rag_context(self, query: str, n_results: int = 3) -> str:
        """Get relevant context from knowledge base for RAG"""
        if not self.rag_enabled:
            return ""
        
        try:
            similar_docs = await self.search_similar_documents(query, n_results)
            
            if not similar_docs:
                return ""
            
            context_parts = []
            for i, doc in enumerate(similar_docs, 1):
                content = doc['content'][:500]  # Limit content length
                metadata = doc.get('metadata', {})
                source = metadata.get('source', 'Unknown')
                title = metadata.get('title', 'Untitled')
                
                context_parts.append(f"""
Document {i} (Source: {source}, Title: {title}, Similarity: {doc['similarity_score']:.3f}):
{content}
""")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"❌ Error getting RAG context: {e}")
            return ""
    
    async def answer_question_with_rag(self, question: str) -> str:
        """Answer a general question using RAG (Retrieval-Augmented Generation)"""
        if not self.initialized:
            raise Exception("Vertex AI service not initialized")
        
        try:
            print(f"🤖 Answering question with RAG: {question}")
            
            # Get relevant context from knowledge base
            rag_context = await self.get_rag_context(question, n_results=5)
            
            # Build RAG prompt
            rag_prompt = f"""
You are an expert assistant for Trellix cybersecurity products and services. Use the following relevant documentation from the knowledge base to answer the user's question accurately and comprehensively.

USER QUESTION: "{question}"

RELEVANT DOCUMENTATION:
{rag_context if rag_context else "No relevant documentation found in knowledge base."}

INSTRUCTIONS:
1. Answer the question using primarily the information from the provided documentation
2. If the documentation doesn't contain enough information, clearly state what information is missing
3. Provide specific, actionable information when possible
4. Include relevant details about Trellix products, features, or configurations mentioned in the documentation
5. If no relevant documentation is found, provide a general response based on your knowledge but clearly indicate this

Please provide a comprehensive and helpful answer:
"""
            
            # Get response from Vertex AI
            print("🧠 Generating RAG response...")
            response = self.model.generate_content(rag_prompt)
            
            if not response or not response.text:
                return "I apologize, but I couldn't generate a response to your question. Please try rephrasing your question."
            
            print(f"✅ Generated RAG response with {len(rag_context)} characters of context")
            return response.text
            
        except Exception as e:
            print(f"❌ Error answering question with RAG: {e}")
            return f"I encountered an error while processing your question: {str(e)}. Please try again."
    
    async def process_user_request(
        self, 
        user_request: str, 
        context: Optional[Dict[str, Any]] = None,
        current_page: Optional[str] = None
    ) -> MCPResponse:
        """
        Process user request and generate action steps
        """
        if not self.initialized:
            raise Exception("Vertex AI service not initialized")
        
        try:
            print(f"🤖 Processing user request: {user_request}")
            print(f"📍 Current page: {current_page}")
            print(f"🔍 Context keys: {list(context.keys()) if context else 'None'}")
            
            # Build the prompt for the LLM with RAG context
            prompt = await self._build_automation_prompt(user_request, context, current_page)
            
            # Get response from Vertex AI
            print("🧠 Sending request to Vertex AI...")
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                raise Exception("Empty response from Vertex AI")
            
            print(f"📝 Raw LLM response length: {len(response.text)} characters")
            
            # Parse the response into action steps
            action_steps = self._parse_llm_response(response.text)
            
            if not action_steps:
                return MCPResponse(
                    success=False,
                    message="Could not generate valid action steps from the request",
                    action_steps=[]
                )
            
            # Calculate confidence score based on various factors
            confidence_score = self._calculate_confidence_score(
                user_request, action_steps, context, current_page
            )
            
            # Generate warnings based on action analysis
            warnings = self._generate_warnings(action_steps, context)
            
            # Calculate total estimated time
            total_time = sum(step.estimated_duration or 1000 for step in action_steps)
            
            # Generate human-readable message
            message = self._generate_response_message(user_request, action_steps, confidence_score)
            
            print(f"✅ Generated {len(action_steps)} action steps with confidence {confidence_score:.2f}")
            
            return MCPResponse(
                success=True,
                message=message,
                action_steps=action_steps,
                estimated_total_time=total_time,
                confidence_score=confidence_score,
                warnings=warnings if warnings else None
            )
            
        except Exception as e:
            print(f"❌ Error processing user request: {e}")
            return MCPResponse(
                success=False,
                message=f"Failed to process request: {str(e)}",
                action_steps=[]
            )
    
    def _calculate_confidence_score(
        self, 
        user_request: str, 
        action_steps: List[ActionStep], 
        context: Optional[Dict[str, Any]], 
        current_page: Optional[str]
    ) -> float:
        """Calculate confidence score for the generated action plan"""
        try:
            score = 0.5  # Base score
            
            # Factor 1: Number of steps (sweet spot is 1-5 steps)
            num_steps = len(action_steps)
            if 1 <= num_steps <= 3:
                score += 0.2
            elif 4 <= num_steps <= 5:
                score += 0.1
            elif num_steps > 10:
                score -= 0.2
            
            # Factor 2: Quality of selectors
            selector_quality = 0
            for step in action_steps:
                if step.target:
                    if '[data-testid=' in step.target or '#' in step.target:
                        selector_quality += 0.1  # High quality
                    elif ':contains(' in step.target or 'button:' in step.target:
                        selector_quality += 0.05  # Medium quality
                    else:
                        selector_quality += 0.01  # Basic quality
            
            score += min(selector_quality, 0.2)
            
            # Factor 3: Action type appropriateness
            action_types = [step.action_type.value for step in action_steps]
            if 'navigate' in action_types and current_page:
                score += 0.1  # Navigation is clear
            if 'click' in action_types:
                score += 0.05  # Common interaction
            if action_types.count('custom') / len(action_types) > 0.5:
                score -= 0.2  # Too many custom actions
            
            # Factor 4: Context availability
            if context and context.get('page_elements'):
                score += 0.1  # Good context
            if context and context.get('current_url'):
                score += 0.05  # URL context
            
            # Factor 5: Request clarity (simple heuristic)
            request_words = user_request.lower().split()
            clear_action_words = ['click', 'navigate', 'go', 'search', 'find', 'add', 'open', 'close']
            if any(word in request_words for word in clear_action_words):
                score += 0.1
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"⚠️ Error calculating confidence score: {e}")
            return 0.5
    
    def _generate_warnings(
        self, 
        action_steps: List[ActionStep], 
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate warnings based on action analysis"""
        warnings = []
        
        try:
            # Check for potentially problematic selectors
            for step in action_steps:
                if step.target:
                    if ':contains(' in step.target and len(step.target.split('"')) > 3:
                        warnings.append(f"Step '{step.id}' uses text-based selector which may be unreliable")
                    
                    if step.target == 'body' or step.target == '*':
                        warnings.append(f"Step '{step.id}' uses very generic selector")
            
            # Check for missing validation steps
            has_navigation = any(step.action_type == ActionType.NAVIGATE for step in action_steps)
            has_validation = any(step.action_type == ActionType.VALIDATE for step in action_steps)
            
            if has_navigation and not has_validation and len(action_steps) > 1:
                warnings.append("Consider adding validation steps after navigation")
            
            # Check for timing issues
            total_time = sum(step.estimated_duration or 1000 for step in action_steps)
            if total_time > 30000:  # More than 30 seconds
                warnings.append(f"Total execution time ({total_time/1000:.1f}s) may be quite long")
            
            # Check for complex workflows
            if len(action_steps) > 8:
                warnings.append("Complex workflow detected - consider breaking into smaller tasks")
            
        except Exception as e:
            print(f"⚠️ Error generating warnings: {e}")
        
        return warnings
    
    def _generate_response_message(
        self, 
        user_request: str, 
        action_steps: List[ActionStep], 
        confidence_score: float
    ) -> str:
        """Generate a human-readable response message"""
        try:
            num_steps = len(action_steps)
            action_types = [step.action_type.value for step in action_steps]
            
            # Primary action description
            if 'navigate' in action_types:
                primary_action = "navigation and interaction"
            elif 'click' in action_types:
                primary_action = "clicking and interaction"
            elif 'type' in action_types:
                primary_action = "text input and interaction"
            else:
                primary_action = "automation"
            
            # Confidence indicator
            if confidence_score >= 0.8:
                confidence_text = "with high confidence"
            elif confidence_score >= 0.6:
                confidence_text = "with good confidence"
            elif confidence_score >= 0.4:
                confidence_text = "with moderate confidence"
            else:
                confidence_text = "with some uncertainty"
            
            message = f"I'll help you {primary_action} {confidence_text}. Generated {num_steps} step{'s' if num_steps != 1 else ''} for: {user_request}"
            
            return message
            
        except Exception as e:
            print(f"⚠️ Error generating response message: {e}")
            return f"Generated {len(action_steps)} action steps for: {user_request}"
    
    async def _build_automation_prompt(
        self, 
        user_request: str, 
        context: Optional[Dict[str, Any]], 
        current_page: Optional[str]
    ) -> str:
        """Build a comprehensive prompt for the LLM with RAG context"""
        
        # Extract comprehensive context information
        page_elements = context.get('page_elements', []) if context else []
        page_title = context.get('page_title', 'Unknown') if context else 'Unknown'
        page_description = context.get('page_description', 'No description') if context else 'No description'
        available_actions = context.get('available_actions', []) if context else []
        navigation_context = context.get('navigation_context', {}) if context else {}
        page_state = context.get('page_state', {}) if context else {}
        page_data = context.get('page_data', {}) if context else {}
        user_intent_context = context.get('user_intent_context', {}) if context else {}
        
        # Get RAG context from knowledge base
        rag_context = await self.get_rag_context(user_request, n_results=3)
        
        # Build enhanced sitemap and context information
        sitemap_info = self._build_enhanced_sitemap_info(context)
        page_structure_info = self._build_page_structure_info(page_elements)
        current_page_analysis = self._build_current_page_analysis(page_data, page_state, user_intent_context)
        
        base_prompt = f"""
You are an expert web automation assistant for a React integration management application. A user wants to perform the following action:

USER REQUEST: "{user_request}"

RELEVANT KNOWLEDGE BASE CONTEXT:
{rag_context if rag_context else "No relevant documentation found in knowledge base."}

CURRENT APPLICATION STATE:
- Current Page: {current_page or "Unknown"}
- Page Title: {page_title}
- Page Description: {page_description}
- Available Page Actions: {', '.join(available_actions)}

CURRENT PAGE ANALYSIS:
{current_page_analysis}

NAVIGATION CONTEXT:
{self._format_navigation_context(navigation_context)}

APPLICATION SITEMAP & STRUCTURE:
{sitemap_info}

CURRENT PAGE ELEMENTS:
{page_structure_info}

ACTION PLANNING GUIDELINES:
Based on the user request and current context, determine the BEST sequence of actions:

1. ASSESS CURRENT STATE: 
   - What page are we on?
   - What information is currently visible?
   - What's the user's likely intent?

2. DETERMINE TARGET STATE:
   - Where does the user want to go?
   - What do they want to accomplish?
   - What information do they need?

3. PLAN OPTIMAL PATH:
   - Can we accomplish this on the current page?
   - Do we need to navigate somewhere first?
   - What's the most efficient sequence?

AVAILABLE ACTION TYPES (use exactly these values):
1. "click" - Click on buttons, links, or clickable elements
2. "type" - Type text into input fields, search boxes, text areas
3. "navigate" - Navigate to different pages within the app (use page names from sitemap)
4. "scroll" - Scroll to specific elements or positions on the page
5. "wait" - Wait for elements to appear or conditions to be met
6. "validate" - Check if elements exist or have specific content
7. "extract" - Extract data/text from elements
8. "custom" - Custom JavaScript execution (use sparingly)

INTELLIGENT SELECTOR STRATEGY:
- For navigation: Use page names directly (e.g., "your-integrations", "available-integrations")
- For buttons: Try these patterns in order:
  1. [data-testid="button-name"] (preferred)
  2. button[aria-label="Button Text"] (accessible)
  3. .btn-class-name or #button-id (specific)
  4. button (then filter by text in MCP client)
- For inputs: input[name="field-name"], input[placeholder="placeholder"], input[type="text"]
- For links: a[href="/path"], a (then filter by text in MCP client)
- AVOID :contains() - use simple selectors and let MCP client handle text matching
- Always prefer data-testid, id, or name attributes when available

CONTEXT-AWARE ACTION PLANNING:
- If current page already has what user needs, work with current elements
- If user wants to "search" and we're not on a search-enabled page, navigate first
- If user wants to "add integration" and we're on available-integrations, help them select one
- If user wants to "go back" from available-integrations, navigate to your-integrations
- Consider current search terms and filter states when planning actions

RESPONSE FORMAT - Return ONLY a valid JSON object:
{{
  "steps": [
    {{
      "id": "step_1",
      "action_type": "navigate|click|type|wait|validate|extract|scroll|custom",
      "description": "Clear, specific description of what this step accomplishes",
      "target": "reliable CSS selector or page name for navigation",
      "value": "text for type actions (optional)",
      "priority": "high|medium|low",
      "estimated_duration": 500,
      "depends_on": ["previous_step_id"],
      "reasoning": "Why this step is necessary given the current context"
    }}
  ]
}}

EXAMPLE INTELLIGENT RESPONSES:
- If user says "search for Slack" and we're on your-integrations page: Navigate to available-integrations first, then search
- If user says "add integration" and we're on your-integrations: Navigate to available-integrations to browse options
- If user says "go back" and we're on available-integrations: Navigate to your-integrations
- If user says "filter by Security" and we're already on the right page: Use existing filter controls

REMEMBER: 
1. Analyze the current context before planning actions
2. Choose the most efficient path to accomplish the user's goal
3. Use the rich context information to make intelligent decisions
4. Provide clear reasoning for each step in complex sequences
"""
        
        return base_prompt
    
    def _build_enhanced_sitemap_info(self, context: Optional[Dict[str, Any]]) -> str:
        """Build enhanced sitemap information for the prompt"""
        sitemap = """
Available Pages:
1. "your-integrations" - Main dashboard showing user's active integrations
   - Features: Integration cards, "Add Integration" button, search functionality
   - Key elements: Integration list, add button, search input
   
2. "available-integrations" - Browse and discover new integrations
   - Features: Integration catalog, search/filter, detailed integration cards
   - Key elements: Integration grid, search bar, filter dropdowns, "Select" buttons
   
3. "add-integration" - Configure and add a new integration
   - Features: Integration setup form, configuration options, save/cancel
   - Key elements: Configuration form, input fields, save/cancel buttons
   
4. "browser-automation" - Demo page for browser automation features
   - Features: Automation controls, demo buttons, test interfaces
   - Key elements: Demo buttons, automation controls
   
5. "llm-demo" - Demo page for LLM integration features
   - Features: LLM testing interface, command input, response display
   - Key elements: Command input, test buttons, response area
   
6. "gemini-test" - Test page for Gemini AI integration
   - Features: Gemini API testing, query interface, response display
   - Key elements: Query input, test buttons, API response display

Navigation: Use the page names as targets for navigate actions (e.g., "your-integrations")
"""
        
        # Add current page context if available
        if context and 'available_pages' in context:
            available_pages = context['available_pages']
            current_page = context.get('current_page', 'Unknown')
            sitemap += f"\nCurrent Page: {current_page}\nAvailable Pages: {', '.join(available_pages)}"
            
        return sitemap
    
    def _build_current_page_analysis(self, page_data: Dict[str, Any], page_state: Dict[str, Any], user_intent_context: Dict[str, Any]) -> str:
        """Build current page analysis for better context awareness"""
        analysis = "CURRENT PAGE ANALYSIS:\n"
        
        if page_data:
            page_type = page_data.get('page_type', 'unknown')
            analysis += f"- Page Type: {page_type}\n"
            
            if page_data.get('items_displayed'):
                analysis += f"- Items Currently Displayed: {page_data['items_displayed']}\n"
            
            if page_data.get('has_search_active'):
                current_search = page_data.get('current_search', '')
                analysis += f"- Active Search: '{current_search}'\n"
                
            if page_data.get('filtered_items') != page_data.get('items_displayed'):
                analysis += f"- Filtered Results: {page_data.get('filtered_items', 0)} of {page_data.get('items_displayed', 0)}\n"
            
            if page_data.get('integration_categories'):
                analysis += f"- Available Categories: {', '.join(page_data['integration_categories'])}\n"
        
        if page_state:
            capabilities = []
            if page_state.get('has_search'): capabilities.append('search')
            if page_state.get('has_filters'): capabilities.append('filtering')
            if page_state.get('has_pagination'): capabilities.append('pagination')
            if page_state.get('has_add_button'): capabilities.append('add items')
            
            if capabilities:
                analysis += f"- Page Capabilities: {', '.join(capabilities)}\n"
                
            if page_state.get('visible_items_count'):
                analysis += f"- Visible Interactive Items: {page_state['visible_items_count']}\n"
        
        if user_intent_context:
            if user_intent_context.get('last_search'):
                analysis += f"- User's Last Search: '{user_intent_context['last_search']}'\n"
            if user_intent_context.get('has_selected_integration'):
                analysis += "- User has selected an integration previously\n"
        
        return analysis
    
    def _format_navigation_context(self, navigation_context: Dict[str, Any]) -> str:
        """Format navigation context information"""
        if not navigation_context:
            return "No navigation context available"
            
        context_info = ""
        current_page = navigation_context.get('current_page_name', 'unknown')
        available_pages = navigation_context.get('available_pages', [])
        
        context_info += f"Current Page: {current_page}\n"
        context_info += f"Available Pages: {', '.join(available_pages)}\n"
        
        if navigation_context.get('search_term'):
            context_info += f"Current Search Term: '{navigation_context['search_term']}'\n"
            
        if navigation_context.get('selected_integration'):
            context_info += f"Selected Integration: {navigation_context['selected_integration']}\n"
            
        return context_info
    
    def _build_sitemap_info(self, context: Optional[Dict[str, Any]]) -> str:
        """Build sitemap information for the prompt (legacy method - use _build_enhanced_sitemap_info)"""
        return self._build_enhanced_sitemap_info(context)
    
    def _build_page_structure_info(self, page_elements: List[Dict[str, Any]]) -> str:
        """Build page structure information from current page elements"""
        if not page_elements:
            return "No page structure information available."
        
        structure_info = "Current Page Elements:\n"
        
        # Group elements by type
        buttons = [el for el in page_elements if el.get('tag') == 'button']
        inputs = [el for el in page_elements if el.get('tag') == 'input']
        links = [el for el in page_elements if el.get('tag') == 'a']
        
        if buttons:
            structure_info += "Buttons:\n"
            for i, btn in enumerate(buttons[:10]):  # Limit to 10 elements
                text = btn.get('text', '').strip()[:50]
                test_id = btn.get('testId')
                element_id = btn.get('id')
                
                selectors = []
                if test_id:
                    selectors.append(f'[data-testid="{test_id}"]')
                if element_id:
                    selectors.append(f'#{element_id}')
                if text:
                    selectors.append(f'button:contains("{text}")')
                
                structure_info += f"  - {text or 'Button'} | Selectors: {', '.join(selectors[:3])}\n"
        
        if inputs:
            structure_info += "Input Fields:\n"
            for i, inp in enumerate(inputs[:10]):
                inp_type = inp.get('type', 'text')
                test_id = inp.get('testId')
                element_id = inp.get('id')
                
                selectors = []
                if test_id:
                    selectors.append(f'[data-testid="{test_id}"]')
                if element_id:
                    selectors.append(f'#{element_id}')
                selectors.append(f'input[type="{inp_type}"]')
                
                structure_info += f"  - {inp_type} input | Selectors: {', '.join(selectors[:3])}\n"
        
        if links:
            structure_info += "Links:\n"
            for i, link in enumerate(links[:10]):
                text = link.get('text', '').strip()[:50]
                
                structure_info += f"  - {text or 'Link'} | Selector: a:contains(\"{text}\")\n"
        
        return structure_info
    
    def _parse_llm_response(self, response_text: str) -> List[ActionStep]:
        """Parse the LLM response into ActionStep objects"""
        try:
            # Clean up the response text
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Remove any leading/trailing whitespace after cleanup
            response_text = response_text.strip()
            
            # Find JSON in the response if it's mixed with other text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            
            print(f"🔍 Parsing LLM response: {response_text[:200]}...")
            
            # Parse JSON
            response_data = json.loads(response_text)
            
            # Extract steps
            steps_data = response_data.get("steps", [])
            
            if not steps_data:
                print("⚠️ No steps found in LLM response")
                return []
            
            action_steps = []
            for i, step_data in enumerate(steps_data):
                try:
                    # Validate action_type
                    action_type_str = step_data.get("action_type", "custom")
                    if action_type_str not in [e.value for e in ActionType]:
                        print(f"⚠️ Invalid action_type '{action_type_str}', defaulting to 'custom'")
                        action_type_str = "custom"
                    
                    # Validate priority
                    priority_str = step_data.get("priority", "medium")
                    if priority_str not in [e.value for e in Priority]:
                        print(f"⚠️ Invalid priority '{priority_str}', defaulting to 'medium'")
                        priority_str = "medium"
                    
                    action_step = ActionStep(
                        id=step_data.get("id", f"step_{i + 1}"),
                        action_type=ActionType(action_type_str),
                        description=step_data.get("description", f"Step {i + 1}"),
                        target=step_data.get("target", ""),
                        value=step_data.get("value"),
                        priority=Priority(priority_str),
                        estimated_duration=step_data.get("estimated_duration", 1000),
                        depends_on=step_data.get("depends_on"),
                        conditions=step_data.get("conditions")
                    )
                    action_steps.append(action_step)
                    print(f"✅ Parsed step {i + 1}: {action_step.action_type.value} - {action_step.description}")
                    
                except Exception as step_error:
                    print(f"❌ Error parsing step {i + 1}: {step_error}")
                    continue
            
            print(f"✅ Successfully parsed {len(action_steps)} action steps")
            return action_steps
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"📝 Response text (first 500 chars): {response_text[:500]}")
            
            # Try to extract action information from text response
            fallback_steps = self._extract_actions_from_text(response_text)
            if fallback_steps:
                print(f"🔄 Extracted {len(fallback_steps)} steps using fallback method")
                return fallback_steps
            
            # Return a default error step
            return [ActionStep(
                id="error_step",
                action_type=ActionType.CUSTOM,
                description="Failed to parse automation steps from AI response. Please try rephrasing your request.",
                target="body",
                priority=Priority.LOW
            )]
        
        except Exception as e:
            print(f"❌ Error parsing LLM response: {e}")
            return []
    
    def _extract_actions_from_text(self, text: str) -> List[ActionStep]:
        """Extract action steps from non-JSON text response (fallback method)"""
        try:
            import re
            
            action_steps = []
            
            # Look for common action patterns in text
            patterns = [
                (r"click (?:on )?(?:the )?(.+)", "click"),
                (r"type (?:the )?(?:text )?[\"'](.+)[\"']", "type"),
                (r"navigate to (?:the )?(.+)", "navigate"),
                (r"search for (?:the )?[\"'](.+)[\"']", "type"),
                (r"go to (?:the )?(.+)", "navigate"),
                (r"find (?:the )?(.+)", "validate"),
            ]
            
            step_id = 1
            for pattern, action_type in patterns:
                matches = re.findall(pattern, text.lower())
                for match in matches:
                    target = match.strip()
                    
                    # Generate appropriate selector based on action type
                    if action_type == "navigate":
                        selector = target.replace(" ", "-")
                    elif action_type == "click":
                        selector = f"button:contains('{target}')"
                    elif action_type == "type":
                        selector = "input[type='text'], input[type='search']"
                    else:
                        selector = f"*:contains('{target}')"
                    
                    action_step = ActionStep(
                        id=f"fallback_step_{step_id}",
                        action_type=ActionType(action_type),
                        description=f"Extracted action: {action_type} {target}",
                        target=selector,
                        value=target if action_type == "type" else None,
                        priority=Priority.MEDIUM,
                        estimated_duration=1000
                    )
                    action_steps.append(action_step)
                    step_id += 1
            
            return action_steps[:5]  # Limit to 5 steps
            
        except Exception as e:
            print(f"❌ Fallback extraction failed: {e}")
            return []
    
    async def validate_action_steps(self, steps: List[ActionStep]) -> Dict[str, Any]:
        """Validate if the proposed action steps are feasible"""
        try:
            # Build validation prompt
            steps_json = json.dumps([step.dict() for step in steps], indent=2)
            
            validation_prompt = f"""
You are a web automation expert. Please validate the following action steps for feasibility and best practices:

ACTION STEPS:
{steps_json}

Please analyze these steps and provide feedback on:
1. Whether the CSS selectors are realistic and specific enough
2. If the step order makes logical sense
3. If there are any missing validation or wait steps
4. If the estimated durations are reasonable
5. Any potential issues or improvements

Respond with a JSON object:
{{
  "valid": true/false,
  "feedback": "Overall assessment",
  "issues": ["list of issues if any"],
  "suggestions": ["list of suggestions for improvement"]
}}
"""
            
            response = self.model.generate_content(validation_prompt)
            validation_result = json.loads(response.text.strip())
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "feedback": f"Validation failed: {str(e)}",
                "issues": ["Failed to validate steps"],
                "suggestions": ["Please review the action steps manually"]
            }
