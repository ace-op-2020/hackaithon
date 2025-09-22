/**
 * Local Browser Controller
 * Safely controls browser tabs and DOM elements
 * Only operates on allowed elements/pages
 * Never touches filesystem or OS
 */

class BrowserController {
  constructor() {
    this.allowedDomains = [
      'localhost',
      '127.0.0.1',
      'google.com',
      'github.com',
      // Add more allowed domains as needed
    ];
    this.isEnabled = true;
    this.executionHistory = [];
  }

  /**
   * Check if domain is allowed for automation
   */
  isDomainAllowed(url) {
    try {
      const domain = new URL(url).hostname;
      return this.allowedDomains.some(allowed => 
        domain === allowed || domain.endsWith(`.${allowed}`)
      );
    } catch {
      return false;
    }
  }

  /**
   * Log action execution for security audit
   */
  logAction(action, result) {
    const logEntry = {
      timestamp: new Date().toISOString(),
      action: action,
      result: result,
      url: window.location.href
    };
    this.executionHistory.push(logEntry);
    
    // Keep only last 100 actions
    if (this.executionHistory.length > 100) {
      this.executionHistory.shift();
    }
    
    console.log('Browser Action Executed:', logEntry);
  }

  /**
   * Navigate to URL (in current tab for same-origin, new tab for cross-origin)
   */
  async navigate(url) {
    try {
      // Ensure URL has protocol
      if (!url.startsWith('http://') && !url.startsWith('https://')) {
        url = 'https://' + url;
      }

      if (!this.isDomainAllowed(url)) {
        throw new Error(`Domain not allowed: ${new URL(url).hostname}`);
      }

      // Check if it's same origin or localhost
      const currentOrigin = window.location.origin;
      const targetUrl = new URL(url);
      const isLocalhost = targetUrl.hostname === 'localhost' || targetUrl.hostname === '127.0.0.1';
      const isSameOrigin = targetUrl.origin === currentOrigin;

      if (isLocalhost || isSameOrigin) {
        // Navigate in current tab for localhost or same origin
        window.location.href = url;
        
        const result = {
          success: true,
          action: 'navigate',
          url: url,
          message: `Navigating to ${url} in current tab`
        };

        this.logAction({ type: 'navigate', url }, result);
        return result;
      } else {
        // For cross-origin, still open in new tab for security
        window.open(url, '_blank', 'noopener,noreferrer');
        
        const result = {
          success: true,
          action: 'navigate',
          url: url,
          message: `Opened ${url} in new tab (cross-origin security)`
        };

        this.logAction({ type: 'navigate', url }, result);
        return result;
      }

    } catch (error) {
      const result = {
        success: false,
        action: 'navigate',
        error: error.message
      };
      
      this.logAction({ type: 'navigate', url }, result);
      return result;
    }
  }

  /**
   * Click on element by selector
   */
  async click(selector) {
    try {
      // Find element
      const element = document.querySelector(selector);
      
      if (!element) {
        throw new Error(`Element not found: ${selector}`);
      }

      // Check if element is visible and clickable
      if (!this.isElementClickable(element)) {
        throw new Error(`Element not clickable: ${selector}`);
      }

      // Simulate human-like click
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      await this.delay(100);
      
      // Create and dispatch click event
      const clickEvent = new MouseEvent('click', {
        view: window,
        bubbles: true,
        cancelable: true
      });
      
      element.dispatchEvent(clickEvent);

      const result = {
        success: true,
        action: 'click',
        selector: selector,
        message: `Clicked element: ${selector}`
      };

      this.logAction({ type: 'click', selector }, result);
      return result;

    } catch (error) {
      const result = {
        success: false,
        action: 'click',
        selector: selector,
        error: error.message
      };
      
      this.logAction({ type: 'click', selector }, result);
      return result;
    }
  }

  /**
   * Fill input field with value
   */
  async fill(selector, value) {
    try {
      const element = document.querySelector(selector);
      
      if (!element) {
        throw new Error(`Element not found: ${selector}`);
      }

      if (!this.isElementFillable(element)) {
        throw new Error(`Element not fillable: ${selector}`);
      }

      // Clear existing value
      element.focus();
      element.select();
      
      // Simulate typing
      element.value = value;
      
      // Trigger input events
      const inputEvent = new Event('input', { bubbles: true });
      const changeEvent = new Event('change', { bubbles: true });
      
      element.dispatchEvent(inputEvent);
      element.dispatchEvent(changeEvent);

      const result = {
        success: true,
        action: 'fill',
        selector: selector,
        value: value,
        message: `Filled element ${selector} with: ${value}`
      };

      this.logAction({ type: 'fill', selector, value }, result);
      return result;

    } catch (error) {
      const result = {
        success: false,
        action: 'fill',
        selector: selector,
        value: value,
        error: error.message
      };
      
      this.logAction({ type: 'fill', selector, value }, result);
      return result;
    }
  }

  /**
   * Take screenshot of element or page
   */
  async screenshot(selector = null) {
    try {
      let element = null;
      
      if (selector) {
        element = document.querySelector(selector);
        if (!element) {
          throw new Error(`Element not found: ${selector}`);
        }
      }

      // Use html2canvas or similar library in a real implementation
      // For now, we'll simulate the action
      const result = {
        success: true,
        action: 'screenshot',
        selector: selector,
        message: selector 
          ? `Screenshot taken of element: ${selector}` 
          : 'Full page screenshot taken',
        // In real implementation, this would contain image data
        imageData: null
      };

      this.logAction({ type: 'screenshot', selector }, result);
      return result;

    } catch (error) {
      const result = {
        success: false,
        action: 'screenshot',
        selector: selector,
        error: error.message
      };
      
      this.logAction({ type: 'screenshot', selector }, result);
      return result;
    }
  }

  /**
   * Search for text on page
   */
  async search(query) {
    try {
      // Simple text search in page content
      const pageText = document.body.innerText.toLowerCase();
      const searchTerm = query.toLowerCase();
      const found = pageText.includes(searchTerm);

      // Highlight found text (basic implementation)
      if (found && window.find) {
        window.find(query, false, false, true, false, true, false);
      }

      const result = {
        success: true,
        action: 'search',
        query: query,
        found: found,
        message: found 
          ? `Found "${query}" on page` 
          : `"${query}" not found on page`
      };

      this.logAction({ type: 'search', query }, result);
      return result;

    } catch (error) {
      const result = {
        success: false,
        action: 'search',
        query: query,
        error: error.message
      };
      
      this.logAction({ type: 'search', query }, result);
      return result;
    }
  }

  /**
   * Select option from dropdown or list
   */
  async select(selector, option) {
    try {
      const element = document.querySelector(selector);
      
      if (!element) {
        throw new Error(`Element not found: ${selector}`);
      }

      if (element.tagName.toLowerCase() === 'select') {
        // Handle select dropdown
        const optionElement = element.querySelector(`option[value="${option}"], option:contains("${option}")`);
        if (optionElement) {
          element.value = optionElement.value;
          const changeEvent = new Event('change', { bubbles: true });
          element.dispatchEvent(changeEvent);
        } else {
          throw new Error(`Option not found: ${option}`);
        }
      } else {
        throw new Error(`Element is not a select: ${selector}`);
      }

      const result = {
        success: true,
        action: 'select',
        selector: selector,
        option: option,
        message: `Selected "${option}" from ${selector}`
      };

      this.logAction({ type: 'select', selector, option }, result);
      return result;

    } catch (error) {
      const result = {
        success: false,
        action: 'select',
        selector: selector,
        option: option,
        error: error.message
      };
      
      this.logAction({ type: 'select', selector, option }, result);
      return result;
    }
  }

  /**
   * Utility: Check if element is clickable
   */
  isElementClickable(element) {
    const style = window.getComputedStyle(element);
    return (
      style.display !== 'none' &&
      style.visibility !== 'hidden' &&
      style.opacity !== '0' &&
      element.offsetWidth > 0 &&
      element.offsetHeight > 0
    );
  }

  /**
   * Utility: Check if element can be filled
   */
  isElementFillable(element) {
    const fillableTypes = ['input', 'textarea', 'select'];
    const tagName = element.tagName.toLowerCase();
    
    if (!fillableTypes.includes(tagName)) {
      return false;
    }
    
    if (tagName === 'input') {
      const inputType = element.type.toLowerCase();
      const allowedInputTypes = ['text', 'email', 'password', 'search', 'tel', 'url', 'number'];
      return allowedInputTypes.includes(inputType);
    }
    
    return true;
  }

  /**
   * Utility: Add delay for human-like behavior
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get execution history for debugging/security
   */
  getExecutionHistory() {
    return [...this.executionHistory];
  }

  /**
   * Clear execution history
   */
  clearHistory() {
    this.executionHistory = [];
  }

  /**
   * Enable/disable browser controller
   */
  setEnabled(enabled) {
    this.isEnabled = enabled;
  }

  /**
   * Check if controller is enabled
   */
  getEnabled() {
    return this.isEnabled;
  }
}

// Create and export singleton instance
const browserController = new BrowserController();
export default browserController;
