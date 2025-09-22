import React, { useState, useCallback } from 'react';
import { useLLMIntegration } from '../hooks/useLLMIntegration';

/**
 * Gemini Integration Test Component
 * 
 * Add this component to your app to test Gemini integration with a nice UI
 * You can temporarily add it to App.js: <GeminiTestComponent />
 */

export const GeminiTestComponent = () => {
  const { processCommand, isProcessing } = useLLMIntegration();
  const [testResults, setTestResults] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentTest, setCurrentTest] = useState('');

  const testCommands = [
    'navigate to llm demo',
    'go to browser automation',
    'show available integrations',
    'go to https://google.com',
    'click on search button',
    'fill email with test@example.com',
    'take a screenshot',
    'unknown command test'
  ];

  const runAllTests = useCallback(async () => {
    setIsRunning(true);
    setTestResults([]);
    
    const results = [];
    
    for (let i = 0; i < testCommands.length; i++) {
      const command = testCommands[i];
      setCurrentTest(`Testing ${i + 1}/${testCommands.length}: ${command}`);
      
      try {
        const startTime = Date.now();
        const result = await processCommand(command);
        const duration = Date.now() - startTime;
        
        results.push({
          command,
          result,
          success: true,
          duration,
          timestamp: new Date().toLocaleTimeString()
        });
        
      } catch (error) {
        results.push({
          command,
          error: error.message,
          success: false,
          timestamp: new Date().toLocaleTimeString()
        });
      }
      
      setTestResults([...results]);
      
      // Small delay between tests
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    setIsRunning(false);
    setCurrentTest('');
  }, [processCommand]);

  const runSingleTest = useCallback(async (command) => {
    try {
      setCurrentTest(`Testing: ${command}`);
      const startTime = Date.now();
      const result = await processCommand(command);
      const duration = Date.now() - startTime;
      
      const newResult = {
        command,
        result,
        success: true,
        duration,
        timestamp: new Date().toLocaleTimeString()
      };
      
      setTestResults(prev => [newResult, ...prev]);
      
    } catch (error) {
      const newResult = {
        command,
        error: error.message,
        success: false,
        timestamp: new Date().toLocaleTimeString()
      };
      
      setTestResults(prev => [newResult, ...prev]);
    } finally {
      setCurrentTest('');
    }
  }, [processCommand]);

  const clearResults = useCallback(() => {
    setTestResults([]);
  }, []);

  const getStats = () => {
    const total = testResults.length;
    const passed = testResults.filter(r => r.success).length;
    const failed = total - passed;
    const appNav = testResults.filter(r => r.success && r.result?.type === 'app-navigate').length;
    const browserActions = testResults.filter(r => r.success && r.result?.type !== 'app-navigate' && r.result?.type !== 'unknown').length;
    
    return { total, passed, failed, appNav, browserActions };
  };

  const stats = getStats();

  return (
    <div className="max-w-4xl mx-auto p-6 bg-gray-800 rounded-lg shadow-lg text-white">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">
          🧪 Gemini Integration Test Suite
        </h2>
        <p className="text-gray-300">
          Test the LLM integration and command parsing functionality
        </p>
      </div>

      {/* Controls */}
      <div className="mb-6 space-y-4">
        <div className="flex space-x-4">
          <button
            onClick={runAllTests}
            disabled={isRunning || isProcessing}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed"
          >
            {isRunning ? 'Running Tests...' : 'Run All Tests'}
          </button>
          
          <button
            onClick={clearResults}
            disabled={isRunning || isProcessing}
            className="bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700 disabled:bg-gray-800 disabled:cursor-not-allowed"
          >
            Clear Results
          </button>
        </div>

        {currentTest && (
          <div className="bg-blue-900 border border-blue-700 p-3 rounded-md">
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400"></div>
              <span className="text-blue-300">{currentTest}</span>
            </div>
          </div>
        )}
      </div>

      {/* Quick Test Buttons */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3 text-gray-200">Quick Tests</h3>
        <div className="grid grid-cols-2 gap-2">
          {testCommands.map((command, index) => (
            <button
              key={index}
              onClick={() => runSingleTest(command)}
              disabled={isRunning || isProcessing}
              className="text-left p-2 bg-gray-700 rounded border border-gray-600 hover:bg-gray-600 text-sm disabled:bg-gray-800 disabled:cursor-not-allowed"
            >
              {command}
            </button>
          ))}
        </div>
      </div>

      {/* Statistics */}
      {testResults.length > 0 && (
        <div className="mb-6 bg-gray-700 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-3 text-gray-200">Test Statistics</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-blue-400">{stats.total}</div>
              <div className="text-sm text-gray-400">Total</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-400">{stats.passed}</div>
              <div className="text-sm text-gray-400">Passed</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-red-400">{stats.failed}</div>
              <div className="text-sm text-gray-400">Failed</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-400">{stats.appNav}</div>
              <div className="text-sm text-gray-400">App Nav</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-yellow-400">{stats.browserActions}</div>
              <div className="text-sm text-gray-400">Browser</div>
            </div>
          </div>
          {stats.total > 0 && (
            <div className="mt-3 text-center">
              <span className="text-lg font-semibold">
                Success Rate: {((stats.passed / stats.total) * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      )}

      {/* Results */}
      {testResults.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3 text-gray-200">
            Test Results ({testResults.length})
          </h3>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {testResults.map((test, index) => (
              <div
                key={index}
                className={`p-3 rounded-md border text-sm ${
                  test.success 
                    ? 'bg-green-900 border-green-700' 
                    : 'bg-red-900 border-red-700'
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="font-medium text-white">
                    {test.command}
                  </div>
                  <div className="text-xs text-gray-400">
                    {test.timestamp}
                    {test.duration && ` (${test.duration}ms)`}
                  </div>
                </div>
                
                {test.success ? (
                  <div className="text-green-300">
                    <div><strong>Action:</strong> {test.result.type}</div>
                    <div><strong>Description:</strong> {test.result.description}</div>
                    {test.result.page && <div><strong>Page:</strong> {test.result.page}</div>}
                    {test.result.url && <div><strong>URL:</strong> {test.result.url}</div>}
                    {test.result.confidence && <div><strong>Confidence:</strong> {test.result.confidence}</div>}
                  </div>
                ) : (
                  <div className="text-red-300">
                    <strong>Error:</strong> {test.error}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="mt-6 p-4 bg-blue-900 rounded-lg border border-blue-700">
        <h3 className="text-sm font-semibold text-blue-300 mb-2">
          💡 Testing Instructions
        </h3>
        <div className="text-sm text-blue-200 space-y-1">
          <p>• <strong>App Navigation:</strong> Commands like "navigate to llm demo" should show app-navigate action</p>
          <p>• <strong>Browser Actions:</strong> Commands like "go to google.com" should show navigate action with URL</p>
          <p>• <strong>Check Console:</strong> Look for detailed logs showing Gemini API calls or fallback parser usage</p>
          <p>• <strong>API Status:</strong> If you see fallback parser messages, the system is working but using pattern matching instead of Gemini</p>
        </div>
      </div>
    </div>
  );
};

export default GeminiTestComponent;
