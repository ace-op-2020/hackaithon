import { useState, useCallback } from 'react';
import googleCloudService from '../services/googleCloudService';

export const useLLMIntegration = () => {
  const [isProcessing, setIsProcessing] = useState(false);

  // Enhanced command parser using Google Cloud Vertex AI with Gemini
  const parseCommandWithLLM = useCallback(async (command) => {
    console.log('🔗 [LLMIntegration] Starting parseCommandWithLLM for:', command);
    setIsProcessing(true);
    try {
      console.log('📞 [LLMIntegration] Calling googleCloudService.parseBrowserCommand...');
      const result = await googleCloudService.parseBrowserCommand(command);
      console.log('✅ [LLMIntegration] Got result from googleCloudService:', result);
      return result;
    } catch (error) {
      console.error('❌ [LLMIntegration] LLM parsing failed:', error);
      console.log('🔄 [LLMIntegration] Falling back to simple parser...');
      // Fallback to simple parser
      const fallbackResult = parseCommand(command);
      console.log('🔄 [LLMIntegration] Fallback result:', fallbackResult);
      return fallbackResult;
    } finally {
      setIsProcessing(false);
    }
  }, []);

  // Simple command parser for demo purposes (fallback)
  const parseCommand = useCallback((command) => {
    const lowerCommand = command.toLowerCase();
    
    // Define common patterns and their corresponding actions
    const patterns = [
      {
        pattern: /(?:go to|navigate to|open|visit)\s+(.+)/i,
        action: 'navigate',
        extract: (match) => ({ url: match[1] })
      },
      {
        pattern: /(?:click|tap|press)\s+(?:on\s+)?(.+)/i,
        action: 'click',
        extract: (match) => ({ selector: match[1] })
      },
      {
        pattern: /(?:fill|type|enter)\s+(.+?)\s+(?:with|as|in)\s+(.+)/i,
        action: 'fill',
        extract: (match) => ({ selector: match[1], value: match[2] })
      },
      {
        pattern: /(?:take|capture)\s+(?:a\s+)?screenshot(?:\s+of\s+(.+))?/i,
        action: 'screenshot',
        extract: (match) => ({ selector: match[1] || null })
      },
      {
        pattern: /(?:search|find)\s+(?:for\s+)?(.+)/i,
        action: 'search',
        extract: (match) => ({ query: match[1] })
      },
      {
        pattern: /(?:select|choose)\s+(.+)/i,
        action: 'select',
        extract: (match) => ({ option: match[1] })
      }
    ];

    // Try to match patterns
    for (const { pattern, action, extract } of patterns) {
      const match = command.match(pattern);
      if (match) {
        return {
          action,
          params: extract(match),
          confidence: 'high'
        };
      }
    }

    // Fallback: try to understand intent
    if (lowerCommand.includes('aws') || lowerCommand.includes('integration')) {
      return {
        action: 'navigate',
        params: { url: 'http://localhost:3000' },
        confidence: 'medium'
      };
    }

    return {
      action: 'unknown',
      params: {},
      confidence: 'low'
    };
  }, []);

  // Convert parsed command to executable action
  const convertToAction = useCallback((parsedCommand) => {
    const { action, params, confidence } = parsedCommand;
    
    switch (action) {
      case 'navigate':
        return {
          type: 'navigate',
          url: params.url,
          description: `Navigate to ${params.url}`
        };
      
      case 'app-navigate':
        return {
          type: 'app-navigate',
          page: params.page,
          description: `Navigate to ${params.page} page`
        };
      
      case 'click':
        return {
          type: 'click',
          selector: params.selector,
          description: `Click on ${params.selector}`
        };
      
      case 'fill':
        return {
          type: 'fill',
          selector: params.selector,
          value: params.value,
          description: `Fill ${params.selector} with ${params.value}`
        };
      
      case 'screenshot':
        return {
          type: 'screenshot',
          selector: params.selector,
          description: `Take screenshot${params.selector ? ` of ${params.selector}` : ''}`
        };
      
      case 'search':
        return {
          type: 'search',
          query: params.query,
          description: `Search for ${params.query}`
        };
      
      case 'select':
        return {
          type: 'select',
          option: params.option,
          description: `Select ${params.option}`
        };
      
      default:
        return {
          type: 'unknown',
          description: 'Command not understood'
        };
    }
  }, []);

  // Main function to process user command
  const processCommand = useCallback(async (command) => {
    console.log('🎯 [ProcessCommand] Starting to process:', command);
    const parsed = await parseCommandWithLLM(command);
    console.log('📋 [ProcessCommand] Parsed result:', parsed);
    const action = convertToAction(parsed);
    console.log('⚡ [ProcessCommand] Converted to action:', action);
    
    const result = {
      ...action,
      originalCommand: command,
      confidence: parsed.confidence
    };
    
    console.log('✅ [ProcessCommand] Final result:', result);
    return result;
  }, [parseCommandWithLLM, convertToAction]);

  return {
    isProcessing,
    processCommand,
    parseCommand,
    parseCommandWithLLM
  };
}; 