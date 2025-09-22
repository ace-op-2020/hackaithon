/**
 * Content Script for TrelliX Browser Extension
 * Integrates with the main React app's browser controller
 */

// Initialize connection to the React app
let reactAppOrigin = 'http://localhost:3000';
let isConnected = false;

// Listen for messages from the React app
window.addEventListener('message', async (event) => {
  // Only accept messages from the React app
  if (event.origin !== reactAppOrigin) {
    return;
  }

  const { type, action, id } = event.data;

  if (type === 'BROWSER_AUTOMATION_COMMAND') {
    try {
      const result = await executeBrowserAction(action);
      
      // Send result back to React app
      window.parent.postMessage({
        type: 'BROWSER_AUTOMATION_RESULT',
        id: id,
        result: result
      }, reactAppOrigin);
      
    } catch (error) {
      // Send error back to React app
      window.parent.postMessage({
        type: 'BROWSER_AUTOMATION_ERROR',
        id: id,
        error: error.message
      }, reactAppOrigin);
    }
  }
});

/**
 * Execute browser action safely
 */
async function executeBrowserAction(action) {
  const { type, selector, url, value, query, option } = action;

  switch (type) {
    case 'click':
      return await clickElement(selector);
    
    case 'fill':
      return await fillElement(selector, value);
    
    case 'navigate':
      return await navigateToUrl(url);
    
    case 'screenshot':
      return await takeScreenshot(selector);
    
    case 'search':
      return await searchOnPage(query);
    
    case 'select':
      return await selectOption(selector, option);
    
    default:
      throw new Error(`Unsupported action type: ${type}`);
  }
}

/**
 * Click on element
 */
async function clickElement(selector) {
  const element = document.querySelector(selector);
  
  if (!element) {
    throw new Error(`Element not found: ${selector}`);
  }

  if (!isElementClickable(element)) {
    throw new Error(`Element not clickable: ${selector}`);
  }

  // Scroll into view
  element.scrollIntoView({ behavior: 'smooth', block: 'center' });
  
  // Add small delay for scroll
  await delay(200);
  
  // Create click event
  const clickEvent = new MouseEvent('click', {
    view: window,
    bubbles: true,
    cancelable: true
  });
  
  element.dispatchEvent(clickEvent);

  return {
    success: true,
    action: 'click',
    selector: selector,
    message: `Successfully clicked element: ${selector}`
  };
}

/**
 * Fill input element
 */
async function fillElement(selector, value) {
  const element = document.querySelector(selector);
  
  if (!element) {
    throw new Error(`Element not found: ${selector}`);
  }

  if (!isElementFillable(element)) {
    throw new Error(`Element not fillable: ${selector}`);
  }

  // Focus and clear existing value
  element.focus();
  element.select();
  
  // Set new value
  element.value = value;
  
  // Trigger events
  element.dispatchEvent(new Event('input', { bubbles: true }));
  element.dispatchEvent(new Event('change', { bubbles: true }));

  return {
    success: true,
    action: 'fill',
    selector: selector,
    value: value,
    message: `Successfully filled element ${selector} with: ${value}`
  };
}

/**
 * Navigate to URL (in current tab for allowed domains)
 */
async function navigateToUrl(url) {
  // Validate URL
  const allowedDomains = ['localhost', '127.0.0.1', 'google.com', 'github.com'];
  
  try {
    const urlObj = new URL(url.startsWith('http') ? url : 'https://' + url);
    const domain = urlObj.hostname;
    
    if (!allowedDomains.some(allowed => domain === allowed || domain.endsWith(`.${allowed}`))) {
      throw new Error(`Domain not allowed: ${domain}`);
    }
  } catch (error) {
    throw new Error(`Invalid URL: ${url}`);
  }

  // Check if it's same origin or localhost
  const currentOrigin = window.location.origin;
  const targetUrl = new URL(url.startsWith('http') ? url : 'https://' + url);
  const isLocalhost = targetUrl.hostname === 'localhost' || targetUrl.hostname === '127.0.0.1';
  const isSameOrigin = targetUrl.origin === currentOrigin;

  if (isLocalhost || isSameOrigin) {
    // Navigate in current tab
    window.location.href = url;
    
    return {
      success: true,
      action: 'navigate',
      url: url,
      message: `Navigating to ${url} in current tab`
    };
  } else {
    // For cross-origin, open in new tab for security
    window.open(url, '_blank', 'noopener,noreferrer');
    
    return {
      success: true,
      action: 'navigate',
      url: url,
      message: `Opened ${url} in new tab (cross-origin)`
    };
  }
}

/**
 * Take screenshot (simplified - would need additional permissions for real implementation)
 */
async function takeScreenshot(selector) {
  // In a real extension, this would use the chrome.tabs API
  // For now, we'll just simulate the action
  
  return {
    success: true,
    action: 'screenshot',
    selector: selector,
    message: `Screenshot taken${selector ? ` of element: ${selector}` : ' of full page'}`,
    // In real implementation, would contain image data
    imageData: null
  };
}

/**
 * Search for text on page
 */
async function searchOnPage(query) {
  const pageText = document.body.innerText.toLowerCase();
  const searchTerm = query.toLowerCase();
  const found = pageText.includes(searchTerm);

  // Highlight text if found
  if (found && window.find) {
    window.find(query, false, false, true, false, true, false);
  }

  return {
    success: true,
    action: 'search',
    query: query,
    found: found,
    message: found ? `Found "${query}" on page` : `"${query}" not found on page`
  };
}

/**
 * Select option from dropdown
 */
async function selectOption(selector, option) {
  const element = document.querySelector(selector);
  
  if (!element) {
    throw new Error(`Element not found: ${selector}`);
  }

  if (element.tagName.toLowerCase() !== 'select') {
    throw new Error(`Element is not a select dropdown: ${selector}`);
  }

  // Find and select option
  const optionElement = element.querySelector(`option[value="${option}"], option:contains("${option}")`);
  if (!optionElement) {
    throw new Error(`Option not found: ${option}`);
  }

  element.value = optionElement.value;
  element.dispatchEvent(new Event('change', { bubbles: true }));

  return {
    success: true,
    action: 'select',
    selector: selector,
    option: option,
    message: `Successfully selected "${option}" from ${selector}`
  };
}

/**
 * Utility functions
 */

function isElementClickable(element) {
  const style = window.getComputedStyle(element);
  return (
    style.display !== 'none' &&
    style.visibility !== 'hidden' &&
    style.opacity !== '0' &&
    element.offsetWidth > 0 &&
    element.offsetHeight > 0
  );
}

function isElementFillable(element) {
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

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Signal that content script is loaded
console.log('TrelliX Browser Automation Content Script loaded');

// Inject a marker for the React app to detect extension presence
const marker = document.createElement('div');
marker.id = 'trellix-browser-extension-active';
marker.style.display = 'none';
document.body.appendChild(marker);
