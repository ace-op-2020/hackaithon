import React, { useRef, useEffect, useCallback, memo } from 'react';
import { MessageCircle, X, Send, Bot, User, Loader } from 'lucide-react';
import QuickActionButton from './QuickActionButton';

const Message = memo(({ message }) => {
  const renderContent = () => {
    switch (message.type) {
      case 'image':
        return (
          <img 
            src={message.content} 
            alt="Screenshot" 
            className="max-w-full h-auto rounded border border-gray-600"
            style={{ maxHeight: '200px' }}
          />
        );
      case 'action':
        return (
          <div className="flex items-center space-x-2">
            <Loader className="w-4 h-4 animate-spin text-blue-400" />
            <span className="text-blue-400">{message.content}</span>
          </div>
        );
      case 'plan':
        return (
          <div className="flex items-center space-x-2">
            <span className="text-purple-400 font-semibold">📋 {message.content}</span>
          </div>
        );
      case 'step':
        return (
          <div className="flex items-start space-x-2 ml-4">
            <span className="text-cyan-400 font-mono text-sm">•</span>
            <span className="text-cyan-300 text-sm">{message.content}</span>
          </div>
        );
      case 'summary':
        return (
          <div className="bg-gray-800 p-3 rounded border-l-4 border-blue-500">
            <pre className="text-blue-300 text-sm whitespace-pre-line">{message.content}</pre>
          </div>
        );
      case 'success':
        return (
          <div className="flex items-center space-x-2">
            <span className="text-green-400 font-semibold">✅ {message.content}</span>
          </div>
        );
      case 'error':
        return (
          <div className="flex items-center space-x-2">
            <span className="text-red-400 font-semibold">❌ {message.content}</span>
          </div>
        );
      case 'warning':
        return (
          <div className="flex items-center space-x-2">
            <span className="text-yellow-400 font-semibold">⚠️ {message.content}</span>
          </div>
        );
      case 'info':
        return (
          <div className="flex items-center space-x-2">
            <span className="text-blue-400">ℹ️ {message.content}</span>
          </div>
        );
      default:
        return <p className="text-sm">{message.content}</p>;
    }
  };

  return (
    <div className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-xs rounded-lg p-3 ${
        message.sender === 'user' 
          ? 'bg-blue-600 text-white' 
          : 'bg-gray-700 text-gray-100'
      }`}>
        <div className="flex items-start space-x-2">
          {message.sender === 'bot' && <Bot className="w-4 h-4 mt-0.5 text-blue-400 flex-shrink-0" />}
          {message.sender === 'user' && <User className="w-4 h-4 mt-0.5 flex-shrink-0" />}
          <div className="flex-1">
            {renderContent()}
            <span className="text-xs opacity-70 block mt-1">
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
});

const FloatingChatbot = ({ 
  isOpen, 
  onToggle, 
  messages, 
  inputMessage, 
  onInputChange, 
  onSendMessage,
  isConnected,
  isLoading,
  isExecuting,
  automationEnabled = true,
  quickActions = {},
  serverInfo = null
}) => {
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleKeyPress = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSendMessage();
    }
  }, [onSendMessage]);

  const handleQuickActionExecute = useCallback(async (actionName) => {
    if (quickActions[actionName]) {
      await quickActions[actionName]();
    }
  }, [quickActions]);

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {!isOpen && (
        <button
          onClick={onToggle}
          className="bg-blue-600 hover:bg-blue-700 text-white rounded-full p-4 shadow-lg transition-transform hover:scale-105"
        >
          <MessageCircle className="w-6 h-6" />
        </button>
      )}

      {isOpen && (
        <div className="bg-gray-800 rounded-lg shadow-2xl w-96 h-96 flex flex-col border border-gray-700">
          {/* Chat Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-700">
            <div className="flex items-center space-x-2">
              <Bot className="w-5 h-5 text-blue-400" />
              <span className="font-semibold">TrelliX Assistant</span>
              {/* Server Connection Status */}
              <div className="flex items-center space-x-1">
                {isConnected ? (
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span className="text-xs text-green-400">MCP Connected</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                    <span className="text-xs text-red-400">MCP Offline</span>
                  </div>
                )}
              </div>
              {/* Automation Status */}
              <div className="flex items-center space-x-1">
                {automationEnabled ? (
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    <span className="text-xs text-blue-400">Auto ON</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                    <span className="text-xs text-gray-400">Auto OFF</span>
                  </div>
                )}
              </div>
            </div>
            <button onClick={onToggle} className="text-gray-400 hover:text-white">
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.length === 0 && (
              <div className="text-gray-400 text-center py-8">
                <Bot className="w-8 h-8 mx-auto mb-2 text-blue-400" />
                <p>Hi! I can help you automate browser tasks.</p>
                <div className="text-sm mt-2">
                  <p className={isConnected ? 'text-green-400' : 'text-red-400'}>
                    MCP Server: {isConnected ? 'Connected' : 'Disconnected'}
                  </p>
                  {serverInfo && (
                    <p className="text-xs text-gray-500 mt-1">
                      {serverInfo.url}
                    </p>
                  )}
                  <p className={automationEnabled ? 'text-blue-400' : 'text-gray-400'}>
                    Automation: {automationEnabled ? 'Enabled' : 'Disabled'}
                  </p>
                </div>
                {(!isConnected || !automationEnabled) && (
                  <p className="text-xs mt-2 text-yellow-400">
                    {!isConnected && 'Connect MCP server to enable AI commands'}
                    {!isConnected && !automationEnabled && ' • '}
                    {!automationEnabled && 'Enable automation to control browser'}
                  </p>
                )}
                <div className="mt-4 text-xs text-gray-500">
                  <p>Try commands like:</p>
                  <p>"Navigate to available integrations"</p>
                  <p>"Click on the search button"</p>
                  <p>"Go to your integrations page"</p>
                </div>
              </div>
            )}
            
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t border-gray-700">
            <div className="flex space-x-2">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => onInputChange(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={!isConnected ? "MCP server offline..." : automationEnabled ? "Type a command..." : "Enable automation to start..."}
                disabled={isExecuting || !automationEnabled || !isConnected}
                className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <button
                onClick={onSendMessage}
                disabled={!inputMessage.trim() || isExecuting || !automationEnabled || !isConnected}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg p-2 transition-colors"
              >
                {isExecuting ? (
                  <Loader className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </button>
            </div>
            
            {/* Quick Actions - Only custom actions from LLM/props, no pre-populated ones */}
            <div className="mt-2 flex flex-wrap gap-1">
              {/* Only show custom quick actions passed from parent component */}
              {Object.entries(quickActions).map(([actionName, actionFn]) => (
                <QuickActionButton 
                  key={actionName}
                  text={actionName} 
                  onClick={() => handleQuickActionExecute(actionName)}
                  disabled={isExecuting || !automationEnabled}
                />
              ))}
              {/* Show message if no custom actions available */}
              {Object.keys(quickActions).length === 0 && (
                <div className="text-xs text-gray-500 italic">
                  Type commands like "navigate to google.com" or "take screenshot"
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default memo(FloatingChatbot); 