import { useState, useCallback } from 'react';
import { callLLMAgent } from '../utils/llmAgent';

export const useAutomation = ({ setCurrentPage, setSearchTerm, setSelectedIntegration }) => {
  const [messages, setMessages] = useState([]);

  const addMessage = useCallback((sender, content) => {
    setMessages(prev => [
      ...prev,
      { id: Date.now(), sender, content, timestamp: new Date() }
    ]);
  }, []);

  // Centralized UI action handler
  const performUIAction = useCallback(async (step) => {
    switch (step) {
      case 'navigate_to_available':
        setCurrentPage('available-integrations');
        break;
      case 'search_integration':
        setSearchTerm('amazon-verified-access');
        break;
      case 'select_integration':
        setSelectedIntegration('amazon-verified-access');
        setCurrentPage('add-integration');
        break;
      case 'fill_form':
        // Optionally auto-fill form data here
        break;
      default:
        break;
    }
  }, [setCurrentPage, setSearchTerm, setSelectedIntegration]);

  // Main command execution: LLM agent + UI actions
  const executeCommand = useCallback(async (command) => {
    addMessage('user', command);
    const steps = await callLLMAgent(command);
    for (const { step, message } of steps) {
      addMessage('bot', `🔄 Starting: ${step.replace('_', ' ')}...`);
      await performUIAction(step);
      await new Promise(res => setTimeout(res, 500));
      addMessage('bot', `✅ ${message}`);
    }
    if (steps.length === 0) {
      addMessage('bot', '❌ Sorry, I could not understand or perform that command.');
    }
  }, [addMessage, performUIAction]);

  return {
    messages,
    addMessage,
    executeCommand
  };
}; 