import React, { useState } from 'react';
import { useBrowserAutomation } from '../hooks/useBrowserAutomation';

export const LLMCommandDemo = () => {
  const [command, setCommand] = useState('');
  const [result, setResult] = useState(null);
  const [executeActions, setExecuteActions] = useState(false);
  const { 
    executeCommand, 
    testCommand,
    isExecuting, 
    automationEnabled,
    toggleAutomation 
  } = useBrowserAutomation();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!command.trim()) return;

    try {
      let processedCommand;
      
      if (executeActions && automationEnabled) {
        // Execute the command with browser automation
        processedCommand = await executeCommand(command);
      } else {
        // Just simulate/test the command
        processedCommand = await testCommand(command);
      }
      
      setResult(processedCommand);
    } catch (error) {
      console.error('Error processing command:', error);
      setResult({ 
        error: error.message, 
        originalCommand: command 
      });
    }
  };

  const exampleCommands = [
    "Navigate to https://google.com",
    "Click on the search button",
    "Fill the search box with 'React tutorial'",
    "Take a screenshot",
    "Search for 'JavaScript'",
    "Select the first option"
  ];

  return (
    <div className="max-w-2xl mx-auto p-6 bg-gray-800 rounded-lg shadow-lg text-white">
      <h2 className="text-2xl font-bold mb-6 text-white">
        LLM Command Parser Demo
      </h2>
      
      {/* Automation Toggle */}
      <div className="mb-4 p-3 bg-gray-700 rounded-lg border border-gray-600">
        <div className="flex items-center justify-between">
          <div>
            <label className="block text-sm font-medium text-gray-200">
              Browser Automation
            </label>
            <p className="text-xs text-gray-400">
              {executeActions ? 'Will execute actions in browser' : 'Will only parse and simulate commands'}
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={executeActions}
                onChange={(e) => setExecuteActions(e.target.checked)}
                disabled={!automationEnabled}
                className="mr-2"
              />
              Execute Actions
            </label>
            <button
              onClick={() => toggleAutomation()}
              className={`px-3 py-1 text-sm rounded ${
                automationEnabled 
                  ? 'bg-green-800 text-green-200 hover:bg-green-700'
                  : 'bg-red-800 text-red-200 hover:bg-red-700'
              }`}
            >
              {automationEnabled ? 'Enabled' : 'Disabled'}
            </button>
          </div>
        </div>
      </div>
      
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-200 mb-2">
            Enter a browser automation command:
          </label>
          <input
            type="text"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            placeholder="e.g., Navigate to google.com"
            className="w-full px-3 py-2 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-white bg-gray-700 placeholder-gray-400"
            disabled={isExecuting}
          />
        </div>
        
        <button
          type="submit"
          disabled={isExecuting || !command.trim()}
          className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isExecuting ? 'Processing...' : 
           executeActions && automationEnabled ? 'Execute Command' : 'Parse Command'}
        </button>
      </form>

      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3 text-gray-200">
          Example Commands:
        </h3>
        <div className="grid grid-cols-1 gap-2">
          {exampleCommands.map((example, index) => (
            <button
              key={index}
              onClick={() => setCommand(example)}
              className="text-left p-2 bg-gray-700 rounded border border-gray-600 hover:bg-gray-600 text-sm text-gray-300"
              disabled={isExecuting}
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {result && (
        <div className="mt-6 p-4 bg-gray-700 rounded-lg">
          <h3 className="text-lg font-semibold mb-3 text-gray-200">
            Parsed Result:
          </h3>
          <pre className="bg-gray-800 text-green-400 p-4 rounded-md text-sm overflow-x-auto">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}

      <div className="mt-6 text-sm text-gray-300">
        <p className="mb-2 text-gray-300">
          <strong>Note:</strong> This demo uses Google Cloud Vertex AI with Gemini 
          to intelligently parse natural language commands into structured browser actions.
        </p>
        <p className="mb-2 text-gray-300">
          The system will first attempt to use the LLM for parsing, and fall back 
          to pattern matching if the API is unavailable.
        </p>
        <p className="text-gray-300">
          <strong>Status:</strong> 
          <span className={isExecuting ? "text-yellow-400" : "text-green-400"}>
            {isExecuting ? " Processing..." : " Ready"}
          </span>
          {executeActions && automationEnabled && (
            <span className="ml-2 text-green-400">| Browser Automation Active</span>
          )}
          {executeActions && !automationEnabled && (
            <span className="ml-2 text-red-400">| Browser Automation Disabled</span>
          )}
        </p>
      </div>
    </div>
  );
};
