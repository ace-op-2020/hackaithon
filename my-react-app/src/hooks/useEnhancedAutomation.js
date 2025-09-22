import { useState, useCallback } from 'react';
import { useLLMIntegration } from './useLLMIntegration';

export const useEnhancedAutomation = () => {
  const [messages, setMessages] = useState([]);
  const [isExecuting, setIsExecuting] = useState(false);
  
  const llm = useLLMIntegration();

  const addMessage = useCallback((sender, content, type = 'text') => {
    setMessages(prev => [
      ...prev,
      { 
        id: Date.now(), 
        sender, 
        content, 
        type,
        timestamp: new Date() 
      }
    ]);
  }, []);

  // Execute a single action
  const executeAction = useCallback(async (action) => {
    const { type, description, ...params } = action;
    
    addMessage('bot', `🔄 ${description}...`, 'action');
    
    try {
      let result;
      
      switch (type) {
        case 'navigate':
          // Replace Kapture navigation with simple state update
          result = { success: true, message: `Navigated to ${params.url}` };
          break;
        case 'click':
          // Replace Kapture click with simulated action
          result = { success: true, message: `Clicked on ${params.selector}` };
          break;
        case 'fill':
          // Replace Kapture fill with simulated action
          result = { success: true, message: `Filled ${params.selector} with ${params.value}` };
          break;
        case 'screenshot':
          // Replace Kapture screenshot with simulated action
          result = { success: true, message: `Screenshot captured${params.selector ? ` of ${params.selector}` : ''}` };
          break;
        case 'search':
          // Replace Kapture search with simulated action
          result = { success: true, message: `Searched for "${params.query}"` };
          break;
        case 'select':
          // Replace Kapture select with simulated action
          result = { success: true, message: `Selected option "${params.option}"` };
          break;
        default:
          result = { success: false, error: `Unknown action type: ${type}` };
      }
      
      if (result.success) {
        addMessage('bot', `✅ ${result.message}`, 'success');
      } else {
        addMessage('bot', `❌ ${result.error || 'Action failed'}`, 'error');
      }
      
      return result;
      
    } catch (error) {
      const errorMessage = `❌ Error executing action: ${error.message}`;
      addMessage('bot', errorMessage, 'error');
      return { success: false, error: error.message };
    }
  }, [addMessage]);

  // Execute a sequence of actions
  const executeSequence = useCallback(async (actions) => {
    setIsExecuting(true);
    
    try {
      for (const action of actions) {
        const result = await executeAction(action);
        if (!result.success) {
          addMessage('bot', `⚠️ Stopping execution due to failure: ${result.error}`, 'warning');
          break;
        }
        // Small delay between actions
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } finally {
      setIsExecuting(false);
    }
  }, [executeAction, addMessage]);

  // Main function to process user command and execute automation
  const executeCommand = useCallback(async (command) => {
    // Remove Kapture connection check
    addMessage('user', command);
    
    try {
      // Parse the command using LLM
      const action = await llm.processCommand(command);
      
      if (action.type === 'unknown') {
        addMessage('bot', '❌ I couldn\'t understand that command. Try being more specific.', 'error');
        return;
      }
      
      // Execute the action
      await executeAction(action);
      
    } catch (error) {
      addMessage('bot', `❌ Error processing command: ${error.message}`, 'error');
    }
  }, [addMessage, llm.processCommand, executeAction]);

  // Quick action shortcuts
  const quickActions = useCallback(() => [
    {
      label: 'Navigate to Home',
      action: { type: 'navigate', description: 'Navigate to home page', url: '/' }
    },
    {
      label: 'Search Integrations',
      action: { type: 'search', description: 'Search for integrations', query: 'aws' }
    },
    {
      label: 'Take Screenshot',
      action: { type: 'screenshot', description: 'Capture current page' }
    }
  ], []);

  return {
    messages,
    addMessage,
    executeAction,
    executeSequence,
    executeCommand,
    quickActions,
    isExecuting,
    // Remove Kapture-related properties
    isConnected: true, // Always connected since we're simulating
    isLoading: false,
    tabs: [],
    currentTabId: null
  };
}; 