import React, { useState, useEffect } from 'react';
import { useBrowserAutomation } from '../hooks/useBrowserAutomation';

export const BrowserAutomationDemo = () => {
  const [command, setCommand] = useState('');
  const [executionMode, setExecutionMode] = useState('execute'); // 'execute', 'simulate', 'validate'
  const [showHistory, setShowHistory] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  
  const {
    executeCommand,
    executeCommandSequence,
    testCommand,
    validateCommand,
    getStatus,
    toggleAutomation,
    clearHistory,
    stopExecution,
    executionHistory,
    lastResult,
    isExecuting,
    automationEnabled
  } = useBrowserAutomation();

  const [status, setStatus] = useState({});

  // Update status periodically
  useEffect(() => {
    const updateStatus = () => setStatus(getStatus());
    updateStatus();
    const interval = setInterval(updateStatus, 1000);
    return () => clearInterval(interval);
  }, [getStatus]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!command.trim() || isExecuting) return;

    try {
      let result;
      
      switch (executionMode) {
        case 'execute':
          result = await executeCommand(command);
          break;
        case 'simulate':
          result = await testCommand(command);
          break;
        case 'validate':
          result = await validateCommand(command);
          break;
        default:
          throw new Error('Invalid execution mode');
      }
      
      console.log('Command result:', result);
      
    } catch (error) {
      console.error('Error executing command:', error);
    }
  };

  const handleSequenceTest = async () => {
    // Simple test sequence that will be parsed by LLM
    const testCommands = [
      "Navigate to https://google.com",
      "Take a screenshot"
    ];
    
    try {
      await executeCommandSequence(testCommands, { simulateOnly: true });
    } catch (error) {
      console.error('Error executing sequence:', error);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-gray-800 rounded-lg shadow-lg text-white">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-white">
          Browser Automation System
        </h2>
        
        <div className="flex items-center space-x-4">
          {/* Status Indicator */}
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${
              isExecuting ? 'bg-yellow-500 animate-pulse' :
              automationEnabled ? 'bg-green-500' : 'bg-red-500'
            }`} />
            <span className="text-sm text-gray-600">
              {isExecuting ? 'Executing...' : 
               automationEnabled ? 'Ready' : 'Disabled'}
            </span>
          </div>
          
          {/* Settings Toggle */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 text-gray-500 hover:text-gray-700 rounded-md hover:bg-gray-100"
          >
            ⚙️
          </button>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg border">
          <h3 className="font-semibold mb-3">Settings</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Automation Status
              </label>
              <button
                onClick={() => toggleAutomation()}
                className={`px-4 py-2 rounded-md text-sm font-medium ${
                  automationEnabled
                    ? 'bg-green-100 text-green-800 hover:bg-green-200'
                    : 'bg-red-100 text-red-800 hover:bg-red-200'
                }`}
              >
                {automationEnabled ? 'Enabled' : 'Disabled'}
              </button>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">
                Actions
              </label>
              <div className="space-x-2">
                <button
                  onClick={clearHistory}
                  className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded"
                >
                  Clear History
                </button>
                <button
                  onClick={stopExecution}
                  className="px-3 py-1 text-sm bg-red-100 text-red-800 hover:bg-red-200 rounded"
                  disabled={!isExecuting}
                >
                  Stop
                </button>
              </div>
            </div>
          </div>

          {/* Status Details */}
          <div className="mt-4 text-xs text-gray-600 bg-white p-3 rounded border">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="font-medium">Browser Controller</div>
                <div>{status.browserControllerEnabled ? '✅ Enabled' : '❌ Disabled'}</div>
              </div>
              <div>
                <div className="font-medium">Execution Queue</div>
                <div>{status.executorStatus?.queueLength || 0} items</div>
              </div>
              <div>
                <div className="font-medium">History</div>
                <div>{status.historyCount || 0} entries</div>
              </div>
              <div>
                <div className="font-medium">LLM Status</div>
                <div>{status.isLLMProcessing ? '🔄 Processing' : '✅ Ready'}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Command Input Form */}
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-200 mb-2">
            Enter a browser automation command:
          </label>
          <div className="flex space-x-2">
            <input
              type="text"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              placeholder="e.g., Navigate to google.com and search for React"
              className="flex-1 px-3 py-2 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-white bg-gray-700 placeholder-gray-400"
              disabled={isExecuting}
            />
            
            {/* Execution Mode Selector */}
            <select
              value={executionMode}
              onChange={(e) => setExecutionMode(e.target.value)}
              className="px-3 py-2 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-white bg-gray-700"
              disabled={isExecuting}
            >
              <option value="execute">Execute</option>
              <option value="simulate">Simulate</option>
              <option value="validate">Validate</option>
            </select>
          </div>
        </div>
        
        <div className="flex space-x-2">
          <button
            type="submit"
            disabled={isExecuting || !command.trim() || !automationEnabled}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {isExecuting ? 'Processing...' : 
             executionMode === 'execute' ? 'Execute Command' :
             executionMode === 'simulate' ? 'Simulate Command' :
             'Validate Command'}
          </button>
          
          <button
            type="button"
            onClick={handleSequenceTest}
            disabled={isExecuting || !automationEnabled}
            className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Test Sequence
          </button>
          
          <button
            type="button"
            onClick={() => setShowHistory(!showHistory)}
            className="bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700"
          >
            {showHistory ? 'Hide History' : 'Show History'}
          </button>
        </div>
      </form>

      {/* Last Result */}
      {lastResult && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-semibold mb-3 text-gray-200">
            Last Result:
          </h3>
          
          <div className={`p-3 rounded-md mb-3 ${
            lastResult.success ? 'bg-green-900 border border-green-700' : 
            lastResult.error ? 'bg-red-900 border border-red-700' :
            'bg-yellow-900 border border-yellow-700'
          }`}>
            <div className="flex items-center space-x-2 mb-2">
              <span className={`text-sm font-medium ${
                lastResult.success ? 'text-green-300' :
                lastResult.error ? 'text-red-300' :
                'text-yellow-300'
              }`}>
                {lastResult.success ? '✅ Success' :
                 lastResult.error ? '❌ Error' :
                 '⚠️ Warning'}
              </span>
              {lastResult.simulated && (
                <span className="text-xs bg-blue-800 text-blue-300 px-2 py-1 rounded">
                  SIMULATED
                </span>
              )}
            </div>
            
            <div className="text-sm text-gray-300">
              <strong>Command:</strong> {lastResult.command}
            </div>
            
            {lastResult.executionResult && (
              <div className="text-sm text-gray-300 mt-1">
                <strong>Result:</strong> {lastResult.executionResult.message || lastResult.executionResult.error}
              </div>
            )}
          </div>
          
          <details className="text-sm">
            <summary className="cursor-pointer font-medium text-gray-400 hover:text-gray-300">
              View Details
            </summary>
            <pre className="bg-gray-800 text-green-400 p-4 rounded-md text-xs overflow-x-auto mt-2">
              {JSON.stringify(lastResult, null, 2)}
            </pre>
          </details>
        </div>
      )}

      {/* Execution History */}
      {showHistory && executionHistory.length > 0 && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3 text-gray-200">
            Execution History ({executionHistory.length}):
          </h3>
          
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {executionHistory.map((entry, index) => (
              <div
                key={index}
                className={`p-3 rounded-md border text-sm ${
                  entry.success ? 'bg-green-900 border-green-700' :
                  entry.error ? 'bg-red-900 border-red-700' :
                  'bg-gray-700 border-gray-600'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="font-medium text-white">{entry.command}</div>
                    {entry.executionResult && (
                      <div className="text-gray-300 mt-1">
                        {entry.executionResult.message || entry.executionResult.error}
                      </div>
                    )}
                    {entry.error && (
                      <div className="text-red-300 mt-1">{entry.error}</div>
                    )}
                  </div>
                  
                  <div className="text-xs text-gray-400 ml-4">
                    {new Date(entry.timestamp).toLocaleTimeString()}
                    {entry.simulated && (
                      <span className="block bg-blue-800 text-blue-300 px-1 rounded mt-1">
                        SIM
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Information Panel */}
      <div className="mt-6 p-4 bg-blue-900 rounded-lg border border-blue-700">
        <h3 className="text-sm font-semibold text-blue-300 mb-2">
          🤖 AI-Powered Browser Automation
        </h3>
        <div className="text-sm text-blue-200 space-y-1">
          <p>
            <strong>Cloud LLM Agent:</strong> Uses Gemini AI to understand your natural language commands and convert them into precise browser actions.
          </p>
          <p>
            <strong>Local Browser Controller:</strong> Safely executes actions in the current tab on allowed domains. Never touches your filesystem or OS.
          </p>
          <p>
            <strong>Intelligent Action Parsing:</strong> All actions are determined by AI analysis of your commands - no pre-programmed responses.
          </p>
          <p className="mt-2">
            <strong>Security:</strong> Domain-restricted, fully logged, and can be disabled instantly. Actions execute in current tab when safe.
          </p>
          <p className="mt-2 text-xs">
            <strong>Try commands like:</strong> "Navigate to google.com", "Take a screenshot", "Click the search button", "Fill the input with 'hello world'"
          </p>
        </div>
      </div>
    </div>
  );
};
