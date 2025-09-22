/**
 * Enhanced Browser Automation Hook
 * Integrates LLM command processing with local browser execution
 * Provides safe, controlled browser automation capabilities
 */

import { useState, useCallback, useEffect } from 'react';
import { useLLMIntegration } from './useLLMIntegration';
import actionExecutor from '../services/actionExecutor';
import browserController from '../services/browserController';

export const useBrowserAutomation = () => {
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionHistory, setExecutionHistory] = useState([]);
  const [lastResult, setLastResult] = useState(null);
  const [automationEnabled, setAutomationEnabled] = useState(true);
  const { isProcessing: isLLMProcessing, processCommand } = useLLMIntegration();

  // Initialize browser controller settings
  useEffect(() => {
    browserController.setEnabled(automationEnabled);
  }, [automationEnabled]);

  /**
   * Execute a natural language command using LLM + Browser Automation
   */
  const executeCommand = useCallback(async (naturalLanguageCommand, options = {}) => {
    if (!naturalLanguageCommand?.trim()) {
      throw new Error('Command is required');
    }

    const {
      executeActions = true,
      simulateOnly = false,
      stopOnError = true,
      delayBetweenActions = 500
    } = options;

    setIsExecuting(true);
    setLastResult(null);

    try {
      console.log('🚀 [BrowserAutomation] Processing natural language command:', naturalLanguageCommand);

      // Step 1: Process command with LLM to get structured actions
      const llmResult = await processCommand(naturalLanguageCommand);
      console.log('🧠 [BrowserAutomation] LLM processed command:', llmResult);

      // Step 2: Execute actions if requested (now simulated)
      let executionResult = null;
      
      if (executeActions && llmResult.type !== 'unknown') {
      // Always simulate since Kapture is removed
      console.log('Simulating action execution (Kapture removed)...');
      executionResult = await actionExecutor.simulateAction(llmResult);
    }

      // Step 3: Combine results
      const combinedResult = {
        command: naturalLanguageCommand,
        llmResult: llmResult,
        executionResult: executionResult,
        timestamp: new Date().toISOString(),
        success: executionResult ? executionResult.success : false,
        simulated: simulateOnly
      };

      // Update history
      setExecutionHistory(prev => [combinedResult, ...prev.slice(0, 49)]); // Keep last 50
      setLastResult(combinedResult);

      return combinedResult;

    } catch (error) {
      const errorResult = {
        command: naturalLanguageCommand,
        error: error.message,
        timestamp: new Date().toISOString(),
        success: false,
        simulated: simulateOnly
      };

      setExecutionHistory(prev => [errorResult, ...prev.slice(0, 49)]);
      setLastResult(errorResult);
      
      throw error;

    } finally {
      setIsExecuting(false);
    }
  }, [processCommand]);

  /**
   * Execute multiple commands in sequence
   */
  const executeCommandSequence = useCallback(async (commands, options = {}) => {
    if (!Array.isArray(commands) || commands.length === 0) {
      throw new Error('Commands array is required');
    }

    const {
      stopOnError = true,
      delayBetweenCommands = 1000,
      simulateOnly = false
    } = options;

    setIsExecuting(true);
    const results = [];

    try {
      for (let i = 0; i < commands.length; i++) {
        const command = commands[i];
        
        try {
          console.log(`Executing command ${i + 1}/${commands.length}: ${command}`);
          
          const result = await executeCommand(command, {
            executeActions: true,
            simulateOnly,
            stopOnError: false // Handle errors at sequence level
          });
          
          results.push(result);

          // Stop sequence if command failed and stopOnError is true
          if (!result.success && stopOnError) {
            console.warn(`Command ${i + 1} failed, stopping sequence`);
            break;
          }

          // Add delay between commands
          if (i < commands.length - 1 && delayBetweenCommands > 0) {
            await new Promise(resolve => setTimeout(resolve, delayBetweenCommands));
          }

        } catch (error) {
          const errorResult = {
            command,
            error: error.message,
            timestamp: new Date().toISOString(),
            success: false,
            sequenceIndex: i
          };
          
          results.push(errorResult);

          if (stopOnError) {
            console.error(`Command ${i + 1} failed, stopping sequence:`, error);
            break;
          }
        }
      }

      return results;

    } finally {
      setIsExecuting(false);
    }
  }, [executeCommand]);

  /**
   * Get browser automation status
   */
  const getStatus = useCallback(() => {
    return {
      isExecuting,
      isLLMProcessing,
      automationEnabled,
      executorStatus: actionExecutor.getStatus(),
      browserControllerEnabled: browserController.getEnabled(),
      historyCount: executionHistory.length
    };
  }, [isExecuting, isLLMProcessing, automationEnabled, executionHistory.length]);

  /**
   * Toggle automation on/off
   */
  const toggleAutomation = useCallback((enabled = null) => {
    const newState = enabled !== null ? enabled : !automationEnabled;
    setAutomationEnabled(newState);
    browserController.setEnabled(newState);
    return newState;
  }, [automationEnabled]);

  /**
   * Clear execution history
   */
  const clearHistory = useCallback(() => {
    setExecutionHistory([]);
    setLastResult(null);
    browserController.clearHistory();
  }, []);

  /**
   * Stop current execution
   */
  const stopExecution = useCallback(() => {
    actionExecutor.stop();
    setIsExecuting(false);
  }, []);

  /**
   * Test a command without executing it
   */
  const testCommand = useCallback(async (command) => {
    return await executeCommand(command, { simulateOnly: true });
  }, [executeCommand]);

  /**
   * Get detailed execution history
   */
  const getExecutionHistory = useCallback(() => {
    return {
      automation: executionHistory,
      browser: browserController.getExecutionHistory(),
      queue: actionExecutor.getQueue()
    };
  }, [executionHistory]);

  /**
   * Queue a command for later execution
   */
  const queueCommand = useCallback(async (command) => {
    const llmResult = await processCommand(command);
    return await actionExecutor.queueAction(llmResult);
  }, [processCommand]);

  /**
   * Process queued commands
   */
  const processQueue = useCallback(async () => {
    setIsExecuting(true);
    try {
      return await actionExecutor.processQueue();
    } finally {
      setIsExecuting(false);
    }
  }, []);

  /**
   * Get available example commands
   */
  const getExampleCommands = useCallback(() => {
    return [
      "Navigate to https://google.com",
      "Click on the search input field",
      "Fill the search box with 'React automation'",
      "Click the search button",
      "Take a screenshot of the results",
      "Search for 'JavaScript' on the current page",
      "Open GitHub in a new tab",
      "Navigate to localhost:3000"
    ];
  }, []);

  /**
   * Validate command before execution
   */
  const validateCommand = useCallback(async (command) => {
    try {
      const llmResult = await processCommand(command);
      const validation = actionExecutor.validateAction(llmResult);
      
      return {
        isValid: validation.isValid && llmResult.type !== 'unknown',
        errors: validation.errors,
        llmResult,
        confidence: llmResult.confidence
      };
    } catch (error) {
      return {
        isValid: false,
        errors: [error.message],
        llmResult: null
      };
    }
  }, [processCommand]);

  return {
    // Main execution functions
    executeCommand,
    executeCommandSequence,
    testCommand,
    validateCommand,
    
    // Queue management
    queueCommand,
    processQueue,
    
    // Status and control
    getStatus,
    toggleAutomation,
    stopExecution,
    
    // History and data
    executionHistory,
    lastResult,
    getExecutionHistory,
    clearHistory,
    
    // State
    isExecuting: isExecuting || isLLMProcessing,
    isLLMProcessing,
    automationEnabled,
    
    // Utilities
    getExampleCommands
  };
};
