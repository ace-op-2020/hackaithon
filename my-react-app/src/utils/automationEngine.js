// Automation Engine
export class AutomationEngine {
  constructor(onUpdate) {
    this.onUpdate = onUpdate;
    this.isExecuting = false;
  }

  async executeCommand(command) {
    if (this.isExecuting) {
      return { success: false, message: "Another command is currently executing" };
    }

    this.isExecuting = true;
    try {
      const action = this.parseCommand(command);
      if (!action) {
        return { success: false, message: "Could not understand the command" };
      }

      const result = await this.performAction(action);
      return result;
    } catch (error) {
      return { success: false, message: `Error: ${error.message}` };
    } finally {
      this.isExecuting = false;
    }
  }

  parseCommand(command) {
    const lowerCommand = command.toLowerCase();
    
    if (lowerCommand.includes('create') && lowerCommand.includes('integration')) {
      if (lowerCommand.includes('aws') || lowerCommand.includes('amazon')) {
        return {
          type: 'create_integration',
          integration: 'amazon-verified-access',
          steps: ['navigate_to_available', 'search_integration', 'select_integration', 'fill_form']
        };
      }
      return {
        type: 'create_integration',
        integration: 'general',
        steps: ['navigate_to_available']
      };
    }

    if (lowerCommand.includes('add') && lowerCommand.includes('integration')) {
      return {
        type: 'add_integration',
        steps: ['click_add_button']
      };
    }

    if (lowerCommand.includes('search')) {
      const searchMatch = command.match(/search\s+(?:for\s+)?['""]?([^'""\n]+)['""]?/i);
      return {
        type: 'search',
        query: searchMatch ? searchMatch[1] : 'aws',
        steps: ['perform_search']
      };
    }

    return null;
  }

  async performAction(action) {
    const { type, steps } = action;
    
    for (const step of steps) {
      this.onUpdate({ type: 'step_start', step, action: type });
      await this.delay(1000);
      
      const stepResult = await this.executeStep(step, action);
      
      this.onUpdate({ 
        type: 'step_complete', 
        step, 
        action: type, 
        success: stepResult.success,
        message: stepResult.message 
      });
      
      if (!stepResult.success) {
        return stepResult;
      }
      
      await this.delay(500);
    }

    return { success: true, message: `Successfully completed ${type}` };
  }

  async executeStep(step, action) {
    switch (step) {
      case 'click_add_button':
        return this.simulateClick('Add New Integration');
      case 'navigate_to_available':
        return this.simulateNavigation('Available Integrations');
      case 'search_integration':
        return this.simulateSearch(action.integration);
      case 'select_integration':
        return this.simulateSelect(action.integration);
      case 'fill_form':
        return this.simulateFillForm();
      case 'perform_search':
        return this.simulateSearch(action.query);
      default:
        return { success: false, message: `Unknown step: ${step}` };
    }
  }

  async simulateClick(element) {
    return { success: true, message: `Clicked on ${element}` };
  }

  async simulateNavigation(page) {
    return { success: true, message: `Navigated to ${page}` };
  }

  async simulateSearch(query) {
    return { success: true, message: `Searched for "${query}"` };
  }

  async simulateSelect(item) {
    return { success: true, message: `Selected ${item}` };
  }

  async simulateFillForm() {
    return { success: true, message: "Filled form with default values" };
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
} 