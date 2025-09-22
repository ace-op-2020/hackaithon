/**
 * Action Executor (Local)
 * Reads JSON instructions from LLM and executes them safely in the browser
 * Provides a safe abstraction layer between LLM commands and browser actions
 */

import browserController from './browserController';

class ActionExecutor {
  constructor() {
    this.isExecuting = false;
    this.executionQueue = [];
    this.maxQueueSize = 10;
    this.executionTimeout = 30000; // 30 seconds
  }

  /**
   * Execute a single action from LLM
   */
  async executeAction(actionData) {
    if (!actionData || typeof actionData !== 'object') {
      throw new Error('Invalid action data');
    }

    const { type, url, selector, value, query, option, description } = actionData;

    if (!type) {
      throw new Error('Action type is required');
    }

    // Check if browser controller is enabled
    if (!browserController.getEnabled()) {
      throw new Error('Browser controller is disabled');
    }

    let result;
    
    switch (type) {
      case 'navigate':
        if (!url) {
          throw new Error('URL is required for navigate action');
        }
        result = await browserController.navigate(url);
        break;

      case 'click':
        if (!selector) {
          throw new Error('Selector is required for click action');
        }
        result = await browserController.click(selector);
        break;

      case 'fill':
        if (!selector || value === undefined) {
          throw new Error('Selector and value are required for fill action');
        }
        result = await browserController.fill(selector, value);
        break;

      case 'screenshot':
        result = await browserController.screenshot(selector);
        break;

      case 'search':
        if (!query) {
          throw new Error('Query is required for search action');
        }
        result = await browserController.search(query);
        break;

      case 'select':
        if (!selector || !option) {
          throw new Error('Selector and option are required for select action');
        }
        result = await browserController.select(selector, option);
        break;

      case 'wait':
        const waitTime = actionData.duration || 1000;
        await new Promise(resolve => setTimeout(resolve, waitTime));
        result = {
          success: true,
          action: 'wait',
          duration: waitTime,
          message: `Waited for ${waitTime}ms`
        };
        break;

      case 'unknown':
        result = {
          success: false,
          action: 'unknown',
          error: 'Unknown or unsupported action type',
          originalAction: actionData
        };
        break;

      default:
        throw new Error(`Unsupported action type: ${type}`);
    }

    // Add metadata to result
    return {
      ...result,
      executedAt: new Date().toISOString(),
      originalAction: actionData,
      description: description || `Execute ${type} action`
    };
  }

  /**
   * Execute a sequence of actions from LLM response
   */
  async executeSequence(actionsArray, options = {}) {
    if (!Array.isArray(actionsArray)) {
      // Handle single action
      return [await this.executeAction(actionsArray)];
    }

    const {
      stopOnError = true,
      delayBetweenActions = 500,
      maxActions = 10
    } = options;

    if (actionsArray.length > maxActions) {
      throw new Error(`Too many actions in sequence. Max allowed: ${maxActions}`);
    }

    const results = [];
    this.isExecuting = true;

    try {
      for (let i = 0; i < actionsArray.length; i++) {
        const action = actionsArray[i];
        
        try {
          console.log(`Executing action ${i + 1}/${actionsArray.length}:`, action);
          
          const result = await Promise.race([
            this.executeAction(action),
            new Promise((_, reject) => 
              setTimeout(() => reject(new Error('Action timeout')), this.executionTimeout)
            )
          ]);
          
          results.push(result);

          // Stop if action failed and stopOnError is true
          if (!result.success && stopOnError) {
            console.warn(`Action ${i + 1} failed, stopping sequence:`, result.error);
            break;
          }

          // Add delay between actions for human-like behavior
          if (i < actionsArray.length - 1 && delayBetweenActions > 0) {
            await new Promise(resolve => setTimeout(resolve, delayBetweenActions));
          }

        } catch (error) {
          const errorResult = {
            success: false,
            action: action.type || 'unknown',
            error: error.message,
            executedAt: new Date().toISOString(),
            originalAction: action
          };
          
          results.push(errorResult);

          if (stopOnError) {
            console.error(`Action ${i + 1} failed, stopping sequence:`, error);
            break;
          }
        }
      }

    } finally {
      this.isExecuting = false;
    }

    return results;
  }

  /**
   * Queue action for later execution
   */
  async queueAction(actionData) {
    if (this.executionQueue.length >= this.maxQueueSize) {
      throw new Error('Execution queue is full');
    }

    const queueItem = {
      id: Date.now().toString(),
      action: actionData,
      queuedAt: new Date().toISOString(),
      status: 'queued'
    };

    this.executionQueue.push(queueItem);
    return queueItem.id;
  }

  /**
   * Process queued actions
   */
  async processQueue() {
    if (this.isExecuting || this.executionQueue.length === 0) {
      return;
    }

    const queuedActions = [...this.executionQueue];
    this.executionQueue = [];

    const results = [];

    for (const queueItem of queuedActions) {
      try {
        queueItem.status = 'executing';
        const result = await this.executeAction(queueItem.action);
        queueItem.status = 'completed';
        queueItem.result = result;
        results.push(queueItem);
      } catch (error) {
        queueItem.status = 'failed';
        queueItem.error = error.message;
        results.push(queueItem);
      }
    }

    return results;
  }

  /**
   * Parse LLM response and execute actions
   */
  async executeLLMResponse(llmResponse) {
    try {
      let actionData;

      // Handle different response formats
      if (typeof llmResponse === 'string') {
        try {
          actionData = JSON.parse(llmResponse);
        } catch {
          // If it's not JSON, treat as a simple command
          actionData = {
            type: 'unknown',
            description: llmResponse,
            error: 'Could not parse LLM response as JSON'
          };
        }
      } else if (typeof llmResponse === 'object') {
        actionData = llmResponse;
      } else {
        throw new Error('Invalid LLM response format');
      }

      // Convert from LLM format to executor format if needed
      const normalizedAction = this.normalizeLLMAction(actionData);

      // Execute the action
      return await this.executeAction(normalizedAction);

    } catch (error) {
      return {
        success: false,
        action: 'parse_error',
        error: error.message,
        originalResponse: llmResponse,
        executedAt: new Date().toISOString()
      };
    }
  }

  /**
   * Normalize LLM action format to executor format
   */
  normalizeLLMAction(llmAction) {
    // Handle different LLM response formats
    if (llmAction.action) {
      // Format: { action: "click", params: { selector: "#button" } }
      return {
        type: llmAction.action,
        ...llmAction.params,
        description: llmAction.description,
        confidence: llmAction.confidence
      };
    } else if (llmAction.type) {
      // Format: { type: "click", selector: "#button" }
      return llmAction;
    } else {
      // Unknown format
      return {
        type: 'unknown',
        error: 'Unrecognized LLM action format',
        originalAction: llmAction
      };
    }
  }

  /**
   * Get current execution status
   */
  getStatus() {
    return {
      isExecuting: this.isExecuting,
      queueLength: this.executionQueue.length,
      browserControllerEnabled: browserController.getEnabled(),
      executionHistory: browserController.getExecutionHistory()
    };
  }

  /**
   * Stop current execution (if possible)
   */
  stop() {
    this.isExecuting = false;
    this.executionQueue = [];
  }

  /**
   * Clear execution queue
   */
  clearQueue() {
    this.executionQueue = [];
  }

  /**
   * Get queued actions
   */
  getQueue() {
    return [...this.executionQueue];
  }

  /**
   * Validate action before execution
   */
  validateAction(action) {
    const errors = [];

    if (!action.type) {
      errors.push('Action type is required');
    }

    const requiredParams = {
      navigate: ['url'],
      click: ['selector'],
      fill: ['selector', 'value'],
      select: ['selector', 'option'],
      search: ['query']
    };

    if (requiredParams[action.type]) {
      for (const param of requiredParams[action.type]) {
        if (!action[param] && action[param] !== 0) {
          errors.push(`${param} is required for ${action.type} action`);
        }
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Test mode - simulate actions without actually executing them
   */
  async simulateAction(actionData) {
    const validation = this.validateAction(actionData);
    
    if (!validation.isValid) {
      return {
        success: false,
        action: actionData.type || 'unknown',
        error: validation.errors.join(', '),
        simulated: true
      };
    }

    // Simulate successful execution
    await new Promise(resolve => setTimeout(resolve, 100));
    
    return {
      success: true,
      action: actionData.type,
      message: `Simulated ${actionData.type} action`,
      simulated: true,
      executedAt: new Date().toISOString(),
      originalAction: actionData
    };
  }
}

// Create and export singleton instance
const actionExecutor = new ActionExecutor();
export default actionExecutor;
