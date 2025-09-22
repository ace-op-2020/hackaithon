/**
 * MCP Client Service
 * Handles communication with the Model Context Protocol server
 */

class MCPClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.sessionId = this.generateSessionId();
    this.lastHealthCheck = 0;
    this.healthCheckPromise = null;
  }

  /**
   * Send a request to the MCP server
   */
  async sendRequest(userRequest, context = {}, currentPage = null) {
    try {
      const payload = {
        user_request: userRequest,
        context: {
          ...context,
          current_url: window.location.href,
          page_title: document.title,
          timestamp: new Date().toISOString(),
          user_agent: navigator.userAgent,
          page_structure: this.getPageStructure() // Add page structure info
        },
        current_page: currentPage || window.location.pathname,
        session_id: this.sessionId,
        user_preferences: this.getUserPreferences()
      };

      console.log('📡 [MCPClient] Sending request to MCP server:', payload);

      const response = await fetch(`${this.baseUrl}/mcp/request`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('✅ [MCPClient] MCP server response:', result);
      
      return result;

    } catch (error) {
      console.error('❌ [MCPClient] MCP request failed:', error);
      throw new Error(`MCP request failed: ${error.message}`);
    }
  }

  /**
   * Validate action steps with the MCP server
   */
  async validateSteps(actionSteps) {
    try {
      const response = await fetch(`${this.baseUrl}/mcp/actions/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(actionSteps)
      });

      if (!response.ok) {
        throw new Error(`Validation failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('❌ [MCPClient] Validation failed:', error);
      return { valid: false, feedback: 'Validation request failed' };
    }
  }

  /**
   * Check MCP server health with debouncing
   */
  async checkHealth() {
    const now = Date.now();
    const timeSinceLastCheck = now - this.lastHealthCheck;
    
    // If a health check is already in progress, return that promise
    if (this.healthCheckPromise) {
      console.log('⏳ [MCPClient] Health check already in progress, returning existing promise');
      return this.healthCheckPromise;
    }
    
    // If we checked recently (within 3 seconds), return cached result
    if (timeSinceLastCheck < 3000) {
      console.log(`🕐 [MCPClient] Health check too recent (${timeSinceLastCheck}ms ago), skipping`);
      return { status: 'cached', message: 'Recent health check result' };
    }
    
    // Start a new health check
    this.healthCheckPromise = this._performHealthCheck();
    
    try {
      const result = await this.healthCheckPromise;
      this.lastHealthCheck = now;
      return result;
    } finally {
      this.healthCheckPromise = null;
    }
  }
  
  /**
   * Perform the actual health check
   */
  async _performHealthCheck() {
    try {
      console.log('🏥 [MCPClient] Performing health check at:', `${this.baseUrl}/health`);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        },
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('✅ [MCPClient] Health check successful:', result);
      return result;
      
    } catch (error) {
      console.error('❌ [MCPClient] Health check failed:', error);
      return { status: 'error', message: error.message };
    }
  }

  /**
   * Execute action steps in the browser
   */
  async executeActionSteps(actionSteps) {
    console.log('🎬 [MCPClient] Executing action steps:', actionSteps);
    
    const results = [];
    
    for (let i = 0; i < actionSteps.length; i++) {
      const step = actionSteps[i];
      console.log(`🎯 [MCPClient] Executing step ${i + 1}/${actionSteps.length}:`, step);
      
      try {
        const result = await this.executeStep(step);
        results.push({ step: step.id, success: true, result });
        
        // Wait for the estimated duration
        if (step.estimated_duration) {
          await this.delay(step.estimated_duration);
        }
        
      } catch (error) {
        console.error(`❌ [MCPClient] Step ${step.id} failed:`, error);
        results.push({ step: step.id, success: false, error: error.message });
        
        // Stop execution on error for critical steps
        if (step.priority === 'high') {
          break;
        }
      }
    }
    
    return results;
  }

  /**
   * Execute a single action step
   */
  async executeStep(step) {
    const { action_type, target, value, description } = step;
    
    console.log(`🎯 [MCPClient] Executing ${action_type} on ${target}`);
    console.log(`📝 [MCPClient] Step description: ${description}`);
    
    // For simple selectors like "button", try to find the best match based on description
    let actualTarget = target;
    if (this.isSimpleSelector(target) && description) {
      const smartTarget = this.findSmartSelector(target, description);
      if (smartTarget) {
        console.log(`🧠 [MCPClient] Using smart selector: ${smartTarget} (was: ${target})`);
        actualTarget = smartTarget;
      }
    }
    
    switch (action_type) {
      case 'click':
        return this.clickElement(actualTarget);
        
      case 'type':
        return this.typeInElement(actualTarget, value);
        
      case 'navigate':
        return this.navigateToPage(actualTarget);
        
      case 'scroll':
        return this.scrollToElement(actualTarget);
        
      case 'wait':
        return this.waitForElement(actualTarget, step.estimated_duration || 3000);
        
      case 'validate':
        return this.validateElement(actualTarget);
        
      case 'extract':
        return this.extractFromElement(actualTarget);
        
      case 'custom':
        console.warn(`🔧 [MCPClient] Custom action: ${description}`);
        return this.executeCustomAction(description, actualTarget, value);
        
      default:
        throw new Error(`Unknown action type: ${action_type}`);
    }
  }
  
  /**
   * Check if a selector is simple (just a tag name or class)
   */
  isSimpleSelector(selector) {
    return /^[a-zA-Z]+$/.test(selector) || // Just tag name like "button"
           /^\.[a-zA-Z-_]+$/.test(selector) || // Just class like ".btn"
           /^#[a-zA-Z-_]+$/.test(selector); // Just ID like "#submit"
  }
  
  /**
   * Find a smart selector based on simple selector and description
   */
  findSmartSelector(baseSelector, description) {
    console.log(`🧠 [MCPClient] Finding smart selector for ${baseSelector} with description: ${description}`);
    
    // Extract key words from description that might match element text
    const keyWords = this.extractKeyWords(description);
    console.log(`🔍 [MCPClient] Key words extracted: ${keyWords.join(', ')}`);
    
    // Try to find elements matching the base selector
    let elements;
    try {
      elements = document.querySelectorAll(baseSelector);
    } catch (error) {
      console.warn(`⚠️ [MCPClient] Could not query ${baseSelector}:`, error);
      return null;
    }
    
    if (elements.length === 0) {
      console.log(`📭 [MCPClient] No elements found for ${baseSelector}`);
      return null;
    }
    
    console.log(`📋 [MCPClient] Found ${elements.length} ${baseSelector} elements`);
    
    // Score each element based on how well it matches the description
    let bestElement = null;
    let bestScore = 0;
    
    for (let i = 0; i < elements.length; i++) {
      const element = elements[i];
      const score = this.scoreElementMatch(element, keyWords, description);
      
      console.log(`📊 [MCPClient] Element ${i}: score=${score}, text="${element.textContent.trim().substring(0, 30)}"`);
      
      if (score > bestScore) {
        bestScore = score;
        bestElement = element;
      }
    }
    
    if (bestElement && bestScore > 0) {
      // Generate a more specific selector for the best match
      const specificSelector = this.generateSpecificSelector(bestElement, baseSelector);
      console.log(`✅ [MCPClient] Best match: ${specificSelector} (score: ${bestScore})`);
      return specificSelector;
    }
    
    console.log(`❌ [MCPClient] No good match found for ${baseSelector}`);
    return null;
  }
  
  /**
   * Extract key words from description that might match UI elements
   */
  extractKeyWords(description) {
    // Remove common action words and focus on nouns/object words
    const stopWords = ['click', 'on', 'the', 'button', 'containing', 'with', 'text', 'link', 'input', 'element'];
    const words = description.toLowerCase()
      .replace(/['"]/g, '') // Remove quotes
      .split(/\s+/)
      .filter(word => word.length > 2 && !stopWords.includes(word));
    
    return [...new Set(words)]; // Remove duplicates
  }
  
  /**
   * Score how well an element matches the description
   */
  scoreElementMatch(element, keyWords, description) {
    let score = 0;
    
    const elementText = element.textContent.trim().toLowerCase();
    const ariaLabel = (element.getAttribute('aria-label') || '').toLowerCase();
    const title = (element.title || '').toLowerCase();
    const value = (element.value || '').toLowerCase();
    const placeholder = (element.placeholder || '').toLowerCase();
    
    // Check each key word against element properties
    for (const word of keyWords) {
      if (elementText.includes(word)) score += 10;
      if (ariaLabel.includes(word)) score += 8;
      if (title.includes(word)) score += 6;
      if (value.includes(word)) score += 5;
      if (placeholder.includes(word)) score += 4;
    }
    
    // Bonus for exact matches
    for (const word of keyWords) {
      if (elementText === word) score += 20;
      if (ariaLabel === word) score += 15;
    }
    
    // Bonus for visible and enabled elements
    if (element.offsetParent !== null) score += 2; // Visible
    if (!element.disabled) score += 1; // Enabled
    
    // Penalty for very long text (probably not a button)
    if (elementText.length > 100) score -= 5;
    
    return score;
  }
  
  /**
   * Generate a specific selector for an element
   */
  generateSpecificSelector(element, baseSelector) {
    // Try to create a unique selector
    if (element.id) {
      return `#${element.id}`;
    }
    
    if (element.getAttribute('data-testid')) {
      return `[data-testid="${element.getAttribute('data-testid')}"]`;
    }
    
    if (element.name) {
      return `${baseSelector}[name="${element.name}"]`;
    }
    
    if (element.className) {
      const firstClass = element.className.split(' ')[0];
      if (firstClass && !firstClass.startsWith('css-')) { // Avoid generated CSS-in-JS classes
        return `${baseSelector}.${firstClass}`;
      }
    }
    
    // Use nth-of-type as last resort
    const siblings = Array.from(element.parentNode.querySelectorAll(baseSelector));
    const index = siblings.indexOf(element);
    if (index >= 0) {
      return `${baseSelector}:nth-of-type(${index + 1})`;
    }
    
    return baseSelector; // Fall back to original
  }

  /**
   * Browser action implementations
   */
  async clickElement(selector) {
    const element = this.findElement(selector);
    if (!element) {
      throw new Error(`Element not found: ${selector}`);
    }
    
    // Ensure element is visible and clickable
    if (element.offsetParent === null) {
      throw new Error(`Element not visible: ${selector}`);
    }
    
    // Scroll into view if needed
    element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    await this.delay(200); // Wait for scroll
    
    // Trigger click event
    element.click();
    
    // Also trigger React events if it's a React component
    element.dispatchEvent(new Event('click', { bubbles: true }));
    
    return { success: true, action: 'click', selector, element_text: element.textContent?.trim() };
  }

  async typeInElement(selector, text) {
    const element = this.findElement(selector);
    if (!element) {
      throw new Error(`Element not found: ${selector}`);
    }
    
    // Focus the element first
    element.focus();
    await this.delay(100);
    
    // Clear existing content
    element.value = '';
    element.dispatchEvent(new Event('input', { bubbles: true }));
    
    // Type the new text
    element.value = text;
    
    // Trigger React events
    element.dispatchEvent(new Event('input', { bubbles: true }));
    element.dispatchEvent(new Event('change', { bubbles: true }));
    
    // Trigger keyboard events for better React compatibility
    element.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    element.dispatchEvent(new KeyboardEvent('keyup', { key: 'Enter', bubbles: true }));
    
    return { success: true, action: 'type', selector, text, element_type: element.type };
  }

  async navigateToPage(pageIdentifier) {
    console.log(`🧭 [MCPClient] Navigating to page: ${pageIdentifier}`);
    
    // List of valid page identifiers for the React app
    const validPages = [
      'your-integrations', 
      'available-integrations', 
      'add-integration',
      'browser-automation', 
      'llm-demo', 
      'gemini-test'
    ];
    
    let targetPage = pageIdentifier;
    
    // Clean up the page identifier
    if (targetPage.startsWith('/')) {
      targetPage = targetPage.substring(1);
    }
    if (targetPage.startsWith('#')) {
      targetPage = targetPage.substring(1);
    }
    
    // Validate page identifier
    if (!validPages.includes(targetPage)) {
      // Try to find a matching page
      const matchingPage = validPages.find(page => 
        page.includes(targetPage.toLowerCase()) || 
        targetPage.toLowerCase().includes(page)
      );
      
      if (matchingPage) {
        targetPage = matchingPage;
        console.log(`🔄 [MCPClient] Mapped ${pageIdentifier} to ${targetPage}`);
      } else {
        throw new Error(`Invalid page identifier: ${pageIdentifier}. Valid pages: ${validPages.join(', ')}`);
      }
    }
    
    // For React apps, we need to trigger the app's navigation system
    // This is typically done by clicking navigation elements or calling the app's navigation function
    
    // Try to find navigation elements first
    const navSelectors = [
      `[data-page="${targetPage}"]`,
      `[href="/${targetPage}"]`,
      `[href="#${targetPage}"]`,
      `button:contains("${targetPage.replace('-', ' ')}")`,
      `a:contains("${targetPage.replace('-', ' ')}")`
    ];
    
    for (const selector of navSelectors) {
      try {
        const navElement = this.findElement(selector);
        if (navElement) {
          console.log(`✅ [MCPClient] Found navigation element: ${selector}`);
          navElement.click();
          await this.delay(500); // Wait for navigation
          return { success: true, action: 'navigate', target_page: targetPage, method: 'navigation_click' };
        }
      } catch (error) {
        continue; // Try next selector
      }
    }
    
    // Fallback: Use history API to update URL and trigger React Router
    const newUrl = `/${targetPage}`;
    window.history.pushState({ page: targetPage }, '', newUrl);
    
    // Trigger a popstate event to notify React Router
    window.dispatchEvent(new PopStateEvent('popstate', { state: { page: targetPage } }));
    
    // Trigger a custom event that the React app might be listening for
    window.dispatchEvent(new CustomEvent('navigation', { 
      detail: { page: targetPage, url: newUrl }
    }));
    
    return { success: true, action: 'navigate', target_page: targetPage, url: newUrl, method: 'history_api' };
  }

  async executeCustomAction(description, target, value) {
    console.log(`🔧 [MCPClient] Executing custom action: ${description}`);
    
    // Try to interpret common custom actions
    const lowercaseDesc = description.toLowerCase();
    
    if (lowercaseDesc.includes('refresh') || lowercaseDesc.includes('reload')) {
      window.location.reload();
      return { success: true, action: 'custom', description: 'Page refreshed' };
    }
    
    if (lowercaseDesc.includes('back') || lowercaseDesc.includes('previous')) {
      window.history.back();
      return { success: true, action: 'custom', description: 'Navigated back' };
    }
    
    if (lowercaseDesc.includes('forward') || lowercaseDesc.includes('next')) {
      window.history.forward();
      return { success: true, action: 'custom', description: 'Navigated forward' };
    }
    
    // If we have a target, try to interact with it
    if (target) {
      try {
        const element = this.findElement(target);
        if (element) {
          // Try basic interaction based on element type
          if (element.tagName.toLowerCase() === 'button' || element.type === 'button') {
            element.click();
            return { success: true, action: 'custom', description: `Clicked ${target}` };
          }
          
          if (element.tagName.toLowerCase() === 'input' && value) {
            element.value = value;
            element.dispatchEvent(new Event('input', { bubbles: true }));
            return { success: true, action: 'custom', description: `Set ${target} to ${value}` };
          }
        }
      } catch (error) {
        console.warn('❌ [MCPClient] Custom action element interaction failed:', error);
      }
    }
    
    // Return success but indicate limited execution
    return { 
      success: true, 
      action: 'custom', 
      description: `Custom action acknowledged: ${description}`,
      warning: 'Custom action interpretation may be limited'
    };
  }

  async scrollToElement(selector) {
    const element = this.findElement(selector);
    if (!element) {
      throw new Error(`Element not found: ${selector}`);
    }
    
    element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return { success: true, action: 'scroll', selector };
  }

  async waitForElement(selector, timeout = 3000) {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      const element = this.findElement(selector);
      if (element) {
        return { success: true, action: 'wait', selector, found: true };
      }
      await this.delay(100);
    }
    
    throw new Error(`Element not found within ${timeout}ms: ${selector}`);
  }

  async validateElement(selector) {
    const element = this.findElement(selector);
    const exists = !!element;
    
    return { 
      success: true, 
      action: 'validate', 
      selector, 
      exists,
      visible: exists ? element.offsetParent !== null : false
    };
  }

  async extractFromElement(selector) {
    const element = this.findElement(selector);
    if (!element) {
      throw new Error(`Element not found: ${selector}`);
    }
    
    return {
      success: true,
      action: 'extract',
      selector,
      text: element.textContent,
      value: element.value || '',
      attributes: Array.from(element.attributes).reduce((acc, attr) => {
        acc[attr.name] = attr.value;
        return acc;
      }, {})
    };
  }

  /**
   * Utility methods
   */
  
  /**
   * Intelligent element finder that handles various selector types
   * including non-standard selectors like :contains()
   */
  findElement(selector) {
    console.log(`🔍 [MCPClient] Finding element with selector: ${selector}`);
    
    // Try standard CSS selector first
    try {
      const element = document.querySelector(selector);
      if (element) {
        console.log(`✅ [MCPClient] Found element with standard selector`);
        return element;
      }
    } catch (error) {
      console.log(`⚠️ [MCPClient] Standard selector failed: ${error.message}`);
    }
    
    // Handle :contains() pseudo-selector
    if (selector.includes(':contains(')) {
      return this.findElementByContains(selector);
    }
    
    // Handle other common LLM-generated selectors
    if (selector.includes('text=')) {
      return this.findElementByText(selector);
    }
    
    // Try intelligent fallbacks based on selector content
    const fallbackElement = this.findElementWithFallbacks(selector);
    if (fallbackElement) {
      return fallbackElement;
    }
    
    console.warn(`❌ [MCPClient] No element found with any method for: ${selector}`);
    return null;
  }
  
  /**
   * Try various fallback strategies to find an element
   */
  findElementWithFallbacks(selector) {
    console.log(`🔄 [MCPClient] Trying fallback strategies for: ${selector}`);
    
    // Extract potential text from selector
    const textMatches = selector.match(/['"](.*?)['"]/g);
    if (textMatches) {
      for (const textMatch of textMatches) {
        const text = textMatch.replace(/['"]/g, '');
        console.log(`🎯 [MCPClient] Trying to find element with text: "${text}"`);
        
        // Try to find by button text
        const button = this.findButtonByText(text);
        if (button) return button;
        
        // Try to find by link text
        const link = this.findLinkByText(text);
        if (link) return link;
        
        // Try to find by any element text
        const anyElement = this.findAnyElementByText(text);
        if (anyElement) return anyElement;
      }
    }
    
    // Try fuzzy matching for common patterns
    return this.findElementFuzzy(selector);
  }
  
  /**
   * Find button by text content
   */
  findButtonByText(text) {
    const buttonSelectors = [
      'button', 
      'input[type="button"]', 
      'input[type="submit"]', 
      '[role="button"]', 
      '.btn', 
      '.button',
      'a.btn',
      'a.button'
    ];
    
    for (const sel of buttonSelectors) {
      const buttons = document.querySelectorAll(sel);
      for (const button of buttons) {
        const buttonText = button.textContent.trim();
        const buttonValue = button.value || '';
        const ariaLabel = button.getAttribute('aria-label') || '';
        
        if (buttonText.toLowerCase().includes(text.toLowerCase()) ||
            buttonValue.toLowerCase().includes(text.toLowerCase()) ||
            ariaLabel.toLowerCase().includes(text.toLowerCase())) {
          console.log(`✅ [MCPClient] Found button by text: "${buttonText}" (${sel})`);
          return button;
        }
      }
    }
    return null;
  }
  
  /**
   * Find link by text content
   */
  findLinkByText(text) {
    const links = document.querySelectorAll('a, [role="link"]');
    for (const link of links) {
      const linkText = link.textContent.trim();
      if (linkText.toLowerCase().includes(text.toLowerCase())) {
        console.log(`✅ [MCPClient] Found link by text: "${linkText}"`);
        return link;
      }
    }
    return null;
  }
  
  /**
   * Find any interactive element by text content
   */
  findAnyElementByText(text) {
    const interactiveSelectors = [
      'button', 'a', 'input', 'select', 'textarea', 
      '[role="button"]', '[role="link"]', '[role="tab"]', '[role="menuitem"]',
      '.clickable', '[onclick]', '[data-testid]'
    ];
    
    for (const sel of interactiveSelectors) {
      const elements = document.querySelectorAll(sel);
      for (const element of elements) {
        const elementText = element.textContent.trim();
        const placeholder = element.placeholder || '';
        const ariaLabel = element.getAttribute('aria-label') || '';
        const title = element.title || '';
        
        if (elementText.toLowerCase().includes(text.toLowerCase()) ||
            placeholder.toLowerCase().includes(text.toLowerCase()) ||
            ariaLabel.toLowerCase().includes(text.toLowerCase()) ||
            title.toLowerCase().includes(text.toLowerCase())) {
          console.log(`✅ [MCPClient] Found interactive element by text: "${elementText}" (${sel})`);
          return element;
        }
      }
    }
    return null;
  }
  
  /**
   * Handle :contains() pseudo-selector
   * Example: "button:contains('Select')" -> find button containing "Select"
   */
  findElementByContains(selector) {
    console.log(`🎯 [MCPClient] Handling :contains() selector: ${selector}`);
    
    // Parse the selector: "tagName:contains('text')" or "tagName:contains("text")"
    const match = selector.match(/^([^:]+):contains\(['"]?([^'"]+)['"]?\)(.*)$/);
    if (!match) {
      console.warn(`❌ [MCPClient] Could not parse :contains() selector: ${selector}`);
      return null;
    }
    
    const [, tagName, containsText, additionalSelector] = match;
    console.log(`🔍 [MCPClient] Looking for ${tagName} containing "${containsText}"`);
    
    // Find all elements of the specified tag
    const baseSelector = tagName + (additionalSelector || '');
    let elements;
    
    try {
      elements = document.querySelectorAll(baseSelector);
    } catch (error) {
      // If the base selector is invalid, try just the tag name
      console.warn(`⚠️ [MCPClient] Base selector "${baseSelector}" failed, trying just tag name`);
      elements = document.querySelectorAll(tagName);
    }
    
    // Search for text content match (case insensitive)
    for (const element of elements) {
      const elementText = element.textContent.trim();
      const elementInnerText = element.innerText?.trim() || '';
      
      // Try exact match first
      if (elementText.toLowerCase() === containsText.toLowerCase() || 
          elementInnerText.toLowerCase() === containsText.toLowerCase()) {
        console.log(`✅ [MCPClient] Found ${tagName} with exact text match: "${elementText}"`);
        return element;
      }
    }
    
    // Try contains match
    for (const element of elements) {
      const elementText = element.textContent.trim();
      const elementInnerText = element.innerText?.trim() || '';
      
      if (elementText.toLowerCase().includes(containsText.toLowerCase()) || 
          elementInnerText.toLowerCase().includes(containsText.toLowerCase())) {
        console.log(`✅ [MCPClient] Found ${tagName} with partial text match: "${elementText}"`);
        return element;
      }
    }
    
    // Try aria-label and title attributes
    for (const element of elements) {
      const ariaLabel = element.getAttribute('aria-label') || '';
      const title = element.getAttribute('title') || '';
      
      if (ariaLabel.toLowerCase().includes(containsText.toLowerCase()) || 
          title.toLowerCase().includes(containsText.toLowerCase())) {
        console.log(`✅ [MCPClient] Found ${tagName} with aria-label/title match`);
        return element;
      }
    }
    
    // Try data attributes that might contain the text
    for (const element of elements) {
      const dataAttrs = Array.from(element.attributes)
        .filter(attr => attr.name.startsWith('data-'))
        .map(attr => attr.value.toLowerCase());
      
      if (dataAttrs.some(value => value.includes(containsText.toLowerCase()))) {
        console.log(`✅ [MCPClient] Found ${tagName} with data attribute match`);
        return element;
      }
    }
    
    console.warn(`❌ [MCPClient] No ${tagName} found containing "${containsText}"`);
    console.log(`🔍 [MCPClient] Searched ${elements.length} ${tagName} elements`);
    
    // Debug: Log what we found
    if (elements.length > 0) {
      console.log('📋 [MCPClient] Available elements:', 
        Array.from(elements).slice(0, 5).map(el => ({
          text: el.textContent.trim().substring(0, 50),
          innerHTML: el.innerHTML.substring(0, 100),
          classes: el.className,
          id: el.id
        }))
      );
    }
    
    return null;
  }
  
  /**
   * Handle text= selectors (common in automation tools)
   * Example: "text=Integrations" -> find element with that text
   */
  findElementByText(selector) {
    const textMatch = selector.match(/text=['"]?([^'"]+)['"]?/);
    if (!textMatch) return null;
    
    const searchText = textMatch[1];
    console.log(`🎯 [MCPClient] Finding element by text: "${searchText}"`);
    
    // Search through common interactive elements
    const interactiveSelectors = [
      'button', 'a', 'input[type="button"]', 'input[type="submit"]', 
      '[role="button"]', '.btn', '.button'
    ];
    
    for (const sel of interactiveSelectors) {
      const elements = document.querySelectorAll(sel);
      for (const element of elements) {
        if (element.textContent.trim().toLowerCase().includes(searchText.toLowerCase())) {
          console.log(`✅ [MCPClient] Found element by text: ${sel}`);
          return element;
        }
      }
    }
    
    return null;
  }
  
  /**
   * Fuzzy matching for common patterns
   */
  findElementFuzzy(selector) {
    console.log(`🔍 [MCPClient] Attempting fuzzy match for: ${selector}`);
    
    // Try removing quotes and special characters
    const cleanSelector = selector.replace(/['"]/g, '').replace(/[()]/g, '');
    
    try {
      const element = document.querySelector(cleanSelector);
      if (element) {
        console.log(`✅ [MCPClient] Found element with cleaned selector: ${cleanSelector}`);
        return element;
      }
    } catch (error) {
      // Ignore errors for invalid selectors
    }
    
    // Try common attribute searches
    const commonAttributes = ['data-testid', 'id', 'class', 'name', 'aria-label'];
    const searchTerm = cleanSelector.toLowerCase();
    
    for (const attr of commonAttributes) {
      try {
        const element = document.querySelector(`[${attr}*="${searchTerm}"]`);
        if (element) {
          console.log(`✅ [MCPClient] Found element by ${attr} containing: ${searchTerm}`);
          return element;
        }
      } catch (error) {
        // Continue to next attribute
      }
    }
    
    console.warn(`❌ [MCPClient] No element found with fuzzy matching for: ${selector}`);
    return null;
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  generateSessionId() {
    const sessionId = localStorage.getItem('mcp_session_id');
    if (sessionId) {
      return sessionId;
    }
    
    const newSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('mcp_session_id', newSessionId);
    return newSessionId;
  }

  getUserPreferences() {
    try {
      return JSON.parse(localStorage.getItem('user_preferences') || '{}');
    } catch {
      return {};
    }
  }

  /**
   * Get page structure to help MCP server generate better selectors
   */
  getPageStructure() {
    try {
      const structure = {
        interactive_elements: [],
        navigation_elements: [],
        form_elements: [],
        buttons: [],
        links: [],
        page_metadata: this.getPageMetadata(),
        current_page_analysis: this.analyzeCurrentPage()
      };

      // Find buttons with their text and attributes
      const buttons = document.querySelectorAll('button, input[type="button"], input[type="submit"], [role="button"], .btn, [class*="button"]');
      buttons.forEach((btn, index) => {
        const buttonInfo = {
          index,
          text: btn.textContent.trim(),
          type: btn.tagName.toLowerCase(),
          id: btn.id || null,
          className: btn.className || null,
          'data-testid': btn.getAttribute('data-testid') || null,
          'aria-label': btn.getAttribute('aria-label') || null,
          visible: btn.offsetParent !== null,
          clickable: !btn.disabled,
          selector_suggestions: this.generateSelectorSuggestions(btn)
        };
        structure.buttons.push(buttonInfo);
        structure.interactive_elements.push(buttonInfo);
      });

      // Find navigation links
      const links = document.querySelectorAll('a, [role="link"]');
      links.forEach((link, index) => {
        const linkInfo = {
          index,
          text: link.textContent.trim(),
          href: link.href || null,
          id: link.id || null,
          className: link.className || null,
          'data-page': link.getAttribute('data-page') || null,
          visible: link.offsetParent !== null,
          selector_suggestions: this.generateSelectorSuggestions(link)
        };
        structure.links.push(linkInfo);
        structure.navigation_elements.push(linkInfo);
      });

      // Find form elements
      const formElements = document.querySelectorAll('input, textarea, select');
      formElements.forEach((element, index) => {
        const formInfo = {
          index,
          type: element.type || element.tagName.toLowerCase(),
          placeholder: element.placeholder || null,
          name: element.name || null,
          id: element.id || null,
          className: element.className || null,
          'aria-label': element.getAttribute('aria-label') || null,
          visible: element.offsetParent !== null,
          enabled: !element.disabled,
          selector_suggestions: this.generateSelectorSuggestions(element)
        };
        structure.form_elements.push(formInfo);
      });

      // Add search elements specifically
      const searchElements = document.querySelectorAll('input[type="search"], input[placeholder*="search" i], [class*="search"], [id*="search"]');
      structure.search_elements = Array.from(searchElements).map((el, index) => ({
        index,
        type: 'search',
        placeholder: el.placeholder || null,
        id: el.id || null,
        className: el.className || null,
        selector_suggestions: this.generateSelectorSuggestions(el)
      }));

      return structure;
    } catch (error) {
      console.warn('❌ [MCPClient] Error getting page structure:', error);
      return { error: 'Could not analyze page structure' };
    }
  }

  /**
   * Get page metadata for better context
   */
  getPageMetadata() {
    return {
      title: document.title,
      url: window.location.href,
      pathname: window.location.pathname,
      hash: window.location.hash,
      search: window.location.search,
      domain: window.location.hostname,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      scroll_position: {
        x: window.scrollX,
        y: window.scrollY
      },
      page_load_time: performance.now(),
      user_agent: navigator.userAgent.substring(0, 100) // Truncated for brevity
    };
  }

  /**
   * Analyze current page to provide context about the app state
   */
  analyzeCurrentPage() {
    try {
      const analysis = {
        apparent_page_type: 'unknown',
        main_content_areas: [],
        interactive_count: 0,
        form_count: 0,
        navigation_count: 0,
        content_indicators: []
      };

      // Analyze page type based on content
      const pageTitle = document.title.toLowerCase();
      const bodyText = document.body.textContent.toLowerCase();

      if (pageTitle.includes('integration') || bodyText.includes('integration')) {
        analysis.apparent_page_type = 'integration_page';
        analysis.content_indicators.push('integration_related');
      }

      if (bodyText.includes('available') && bodyText.includes('integration')) {
        analysis.apparent_page_type = 'integration_catalog';
        analysis.content_indicators.push('catalog_view');
      }

      if (bodyText.includes('your') && bodyText.includes('integration')) {
        analysis.apparent_page_type = 'user_integrations';
        analysis.content_indicators.push('user_dashboard');
      }

      if (bodyText.includes('demo') || bodyText.includes('test')) {
        analysis.apparent_page_type = 'demo_page';
        analysis.content_indicators.push('demo_environment');
      }

      // Count interactive elements
      analysis.interactive_count = document.querySelectorAll('button, [role="button"], a, input[type="button"], input[type="submit"]').length;
      analysis.form_count = document.querySelectorAll('form, input, textarea, select').length;
      analysis.navigation_count = document.querySelectorAll('nav, .nav, [role="navigation"], a[href]').length;

      // Identify main content areas
      const mainElements = document.querySelectorAll('main, .main, #main, .content, #content, [role="main"]');
      analysis.main_content_areas = Array.from(mainElements).map(el => ({
        tag: el.tagName.toLowerCase(),
        id: el.id || null,
        className: el.className || null,
        visible: el.offsetParent !== null
      }));

      return analysis;
    } catch (error) {
      console.warn('❌ [MCPClient] Error analyzing current page:', error);
      return { error: 'Could not analyze current page' };
    }
  }

  /**
   * Generate reliable selector suggestions for an element
   */
  generateSelectorSuggestions(element) {
    const suggestions = [];

    // ID selector (most reliable)
    if (element.id) {
      suggestions.push(`#${element.id}`);
    }

    // Data-testid (common in modern apps)
    if (element.getAttribute('data-testid')) {
      suggestions.push(`[data-testid="${element.getAttribute('data-testid')}"]`);
    }

    // Name attribute (for form elements)
    if (element.name) {
      suggestions.push(`[name="${element.name}"]`);
    }

    // Class-based selector (if not too generic)
    if (element.className && !element.className.includes('css-')) {
      const classes = element.className.split(' ').filter(cls => 
        cls.length > 2 && !cls.match(/^(css-|MuiButton|btn-\d)/)
      );
      if (classes.length > 0) {
        suggestions.push(`.${classes[0]}`);
      }
    }

    // Attribute-based selectors
    if (element.getAttribute('aria-label')) {
      suggestions.push(`[aria-label="${element.getAttribute('aria-label')}"]`);
    }

    // Text-based selector (for buttons and links)
    const text = element.textContent.trim();
    if (text && text.length > 0 && text.length < 50) {
      suggestions.push(`${element.tagName.toLowerCase()}:contains("${text}")`);
      suggestions.push(`text="${text}"`); // Alternative format
    }

    // Placeholder-based (for inputs)
    if (element.placeholder) {
      suggestions.push(`[placeholder="${element.placeholder}"]`);
    }

    return suggestions;
  }

  /**
   * Update base URL for the MCP server
   */
  setBaseUrl(baseUrl) {
    this.baseUrl = baseUrl;
  }
}

export default MCPClient;
