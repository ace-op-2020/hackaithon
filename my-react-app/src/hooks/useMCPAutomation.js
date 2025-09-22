/**
 * MCP Browser Automation Hook
 * Integrates with MCP server for intelligent command processing
 */

import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import MCPClient from '../services/mcpClient';

export const useMCPAutomation = (mcpServerUrl = 'http://localhost:8000') => {
  const [isExecuting, setIsExecuting] = useState(false);
  const [isConnected, setIsConnected] = useState(true); // Start as connected, update based on actual usage
  const [executionHistory, setExecutionHistory] = useState([]);
  const [lastResult, setLastResult] = useState(null);
  const [automationEnabled, setAutomationEnabled] = useState(true);
  // Create MCP client instance
  const mcpClient = useMemo(() => new MCPClient(mcpServerUrl), [mcpServerUrl]);

  /**
   * Get detailed page description based on current page
   */
  const getPageDescription = (currentPage) => {
    const pageDescriptions = {
      'your-integrations': 'Page showing user\'s configured integrations with options to manage, filter, and configure existing integrations',
      'available-integrations': 'Catalog page showing all available integrations that can be added, with search and filtering capabilities',
      'llm-demo': 'Demo page for LLM command interface and browser automation capabilities',
      'browser-automation': 'Page for testing browser automation and element interaction features',
      'gemini-test': 'Testing page for Gemini AI integration and capabilities'
    };
    return pageDescriptions[currentPage] || 'Unknown page';
  };

  /**
   * Get available actions based on current page
   */
  const getAvailableActions = (currentPage) => {
    const pageActions = {
      'your-integrations': ['filter integrations', 'search integrations', 'add integration', 'configure integration', 'view integration details'],
      'available-integrations': ['search available integrations', 'filter by category', 'select integration', 'view integration details', 'go back to your integrations'],
      'llm-demo': ['send LLM commands', 'test automation', 'clear chat', 'toggle automation'],
      'browser-automation': ['test browser automation', 'interact with elements', 'take screenshots', 'navigate pages'],
      'gemini-test': ['test Gemini AI', 'send queries', 'view responses', 'test capabilities']
    };
    return pageActions[currentPage] || ['navigate to other pages'];
  };

  /**
   * Get detailed information about page elements
   */
  const getDetailedPageElements = () => {
    const elements = [];
    
    // Get all interactive elements
    const interactiveSelectors = [
      'button', 'input', 'select', 'textarea', 'a[href]',
      '[data-testid]', '[role="button"]', '[role="link"]',
      '[class*="button"]', '[class*="btn"]', '[class*="card"]',
      '[class*="item"]', '[class*="row"]'
    ];
    
    interactiveSelectors.forEach(selector => {
      const foundElements = document.querySelectorAll(selector);
      foundElements.forEach((el, index) => {
        if (index < 15) { // Limit to prevent too much data
          const elementInfo = {
            tag: el.tagName.toLowerCase(),
            id: el.id || null,
            className: el.className || null,
            testId: el.getAttribute('data-testid') || null,
            text: el.textContent?.trim().substring(0, 80) || '',
            type: el.type || null,
            href: el.href || null,
            placeholder: el.placeholder || null,
            ariaLabel: el.getAttribute('aria-label') || null,
            role: el.getAttribute('role') || null,
            visible: el.offsetWidth > 0 && el.offsetHeight > 0,
            position: {
              x: Math.round(el.getBoundingClientRect().x),
              y: Math.round(el.getBoundingClientRect().y)
            }
          };
          
          // Only add if it has meaningful content or interaction capability
          if (elementInfo.text || elementInfo.id || elementInfo.testId || elementInfo.placeholder || elementInfo.ariaLabel) {
            elements.push(elementInfo);
          }
        }
      });
    });
    
    return elements.slice(0, 25); // Limit total elements
  };

  /**
   * Execute a natural language command using MCP server
   */
  const executeCommand = useCallback(async (naturalLanguageCommand, options = {}) => {
    if (!naturalLanguageCommand?.trim()) {
      throw new Error('Command is required');
    }

    const {
      executeActions = true,
      context = {}
    } = options;

    setIsExecuting(true);
    setLastResult(null);

    try {
      console.log('🚀 [MCPAutomation] Processing command:', naturalLanguageCommand);

      // Get comprehensive page context
      const currentPageContext = {
        ...context,
        current_page: window.location.pathname,
        page_title: document.title,
        page_description: getPageDescription(context.current_page),
        available_actions: getAvailableActions(context.current_page),
        page_elements: getDetailedPageElements(),
        navigation_context: {
          available_pages: context.available_pages || [],
          current_page_name: context.current_page || 'unknown',
          search_term: context.search_term || '',
          selected_integration: context.selected_integration || null
        },
        page_state: {
          has_search: !!document.querySelector('input[type="text"], input[placeholder*="search"], input[placeholder*="Search"]'),
          has_filters: !!document.querySelector('[class*="filter"], [data-testid*="filter"]'),
          has_pagination: !!document.querySelector('[class*="pagination"], [aria-label*="pagination"]'),
          has_add_button: !!document.querySelector('button[class*="add"], [data-testid*="add"]') || 
                          Array.from(document.querySelectorAll('button')).some(btn => 
                            btn.textContent.includes('Add') || btn.textContent.includes('+')
                          ),
          visible_items_count: document.querySelectorAll('[class*="card"], [class*="row"], [class*="item"]').length
        }
      };

      // Step 1: Send request to MCP server
      const mcpResponse = await mcpClient.sendRequest(
        naturalLanguageCommand,
        currentPageContext,
        window.location.pathname
      );

      console.log('📨 [MCPAutomation] MCP server response:', mcpResponse);
      
      // Update connection status to true if we got a response
      setIsConnected(true);

      if (!mcpResponse.success) {
        throw new Error(mcpResponse.message || 'MCP server returned error');
      }

      const { action_steps, message, confidence_score, warnings } = mcpResponse;

      // Step 2: Execute actions if requested and available
      let executionResult = null;
      
      if (executeActions && action_steps && action_steps.length > 0) {
        console.log(`🎬 [MCPAutomation] Executing ${action_steps.length} action steps`);
        
        // Show warnings if any (removed popup confirmation)
        if (warnings && warnings.length > 0) {
          console.warn('⚠️ [MCPAutomation] Warnings:', warnings);
          // Warnings will be displayed in chat instead of popup
        }

        // Execute the action steps
        executionResult = await mcpClient.executeActionSteps(action_steps);
        
        console.log('✅ [MCPAutomation] Execution completed:', executionResult);
      }

      // Prepare result
      const result = {
        success: true,
        llmResult: {
          type: mcpResponse.action_steps?.[0]?.action_type || 'unknown',
          description: message,
          confidence: confidence_score,
          actionSteps: action_steps,
          warnings
        },
        executionResult: executionResult ? {
          message: `Executed ${executionResult.filter(r => r.success).length}/${executionResult.length} steps successfully`,
          results: executionResult,
          success: executionResult.every(r => r.success)
        } : null
      };

      // Add to history
      const historyEntry = {
        id: Date.now(),
        command: naturalLanguageCommand,
        result,
        timestamp: new Date().toISOString(),
        executed: executeActions
      };

      setExecutionHistory(prev => [historyEntry, ...prev.slice(0, 49)]); // Keep last 50
      setLastResult(result);

      return result;

    } catch (error) {
      console.error('❌ [MCPAutomation] Command execution failed:', error);
      
      // Update connection status if it's a connection error
      if (error.message.toLowerCase().includes('fetch') || 
          error.message.toLowerCase().includes('network') ||
          error.message.toLowerCase().includes('connection') ||
          error.message.toLowerCase().includes('server')) {
        setIsConnected(false);
      }
      
      const errorResult = {
        success: false,
        error: error.message,
        llmResult: null,
        executionResult: null
      };

      setLastResult(errorResult);
      throw error;

    } finally {
      setIsExecuting(false);
    }
  }, [mcpClient]);

  /**
   * Toggle automation enabled state
   */
  const toggleAutomation = useCallback(() => {
    setAutomationEnabled(prev => !prev);
    return !automationEnabled;
  }, [automationEnabled]);

  /**
   * Clear execution history
   */
  const clearHistory = useCallback(() => {
    setExecutionHistory([]);
  }, []);

  /**
   * Update MCP server URL
   */
  const updateServerUrl = useCallback((newUrl) => {
    mcpClient.setBaseUrl(newUrl);
    // Reset connection status - it will be updated on next command execution
    setIsConnected(true);
  }, [mcpClient]);

  /**
   * Get connection status with server details
   */
  const getConnectionStatus = useCallback(async () => {
    return {
      connected: isConnected,
      serverUrl: mcpClient.baseUrl,
      lastChecked: Date.now() // Current time since we no longer track health checks
    };
  }, [mcpClient, isConnected]);

  return {
    // Core functionality
    executeCommand,
    
    // State
    isExecuting,
    isConnected,
    automationEnabled,
    lastResult,
    executionHistory,
    
    // Controls
    toggleAutomation,
    clearHistory,
    updateServerUrl,
    getConnectionStatus,
    
    // Server info
    serverUrl: mcpClient.baseUrl
  };
};
