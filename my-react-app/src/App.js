import React, { useState, useCallback, useMemo } from 'react';
import Header from './components/Header';
import FloatingChatbot from './components/FloatingChatbot';
import YourIntegrationsPage from './pages/YourIntegrationsPage';
import AvailableIntegrationsPage from './pages/AvailableIntegrationsPage';
import AddIntegrationPage from './pages/AddIntegrationPage';
import { BrowserAutomationDemo } from './components/BrowserAutomationDemo';
import { LLMCommandDemo } from './components/LLMCommandDemo';
import GeminiTestComponent from './components/GeminiTestComponent';
import { mockIntegrations, availableIntegrations } from './data/mockData';
import { useMCPAutomation } from './hooks/useMCPAutomation';

const App = () => {
  const [currentPage, setCurrentPage] = useState('your-integrations');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIntegration, setSelectedIntegration] = useState(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [chatMessages, setChatMessages] = useState([
    {
      id: 0,
      sender: 'bot',
      content: '👋 Hello! I\'m your AI assistant. I can help you navigate and interact with this integration hub. When you give me a command, I\'ll:\n\n1. 📋 Plan the steps needed\n2. 🔍 Show you what I\'ll do\n3. ⚡ Execute the actions\n\nTry asking me to "go to available integrations" or "search for Slack".',
      timestamp: new Date(),
      type: 'info'
    }
  ]);
  
  // MCP Server configuration - make this configurable
  const [mcpServerUrl] = useState(process.env.REACT_APP_MCP_SERVER_URL || 'http://localhost:8000');

  /**
   * Get current page specific data for context
   */
  const getCurrentPageData = useCallback((currentPage, integrations, availableIntegrations, searchTerm, selectedIntegration) => {
    switch (currentPage) {
      case 'your-integrations':
        return {
          page_type: 'user_integrations_list',
          items_displayed: integrations?.length || 0,
          filtered_items: searchTerm ? integrations?.filter(i => 
            i.name.toLowerCase().includes(searchTerm.toLowerCase())
          ).length : integrations?.length || 0,
          has_search_active: !!searchTerm,
          current_search: searchTerm,
          available_actions: ['filter', 'search', 'add integration', 'configure', 'manage'],
          integration_categories: [...new Set(integrations?.map(i => i.category) || [])],
          integration_statuses: [...new Set(integrations?.map(i => i.status) || [])]
        };
        
      case 'available-integrations':
        return {
          page_type: 'available_integrations_catalog',
          items_displayed: availableIntegrations?.length || 0,
          filtered_items: searchTerm ? availableIntegrations?.filter(i => 
            i.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            i.description.toLowerCase().includes(searchTerm.toLowerCase())
          ).length : availableIntegrations?.length || 0,
          has_search_active: !!searchTerm,
          current_search: searchTerm,
          available_actions: ['search', 'filter by category', 'select integration', 'view details', 'go back'],
          integration_categories: [...new Set(availableIntegrations?.map(i => i.category) || [])],
          can_go_back: true
        };
        
      case 'llm-demo':
        return {
          page_type: 'llm_demo_interface',
          available_actions: ['send commands', 'test automation', 'clear chat', 'toggle automation'],
          has_automation: true,
          can_interact_with_elements: true
        };
        
      case 'browser-automation':
        return {
          page_type: 'browser_automation_demo',
          available_actions: ['test browser automation', 'interact with elements', 'take screenshots'],
          has_automation: true,
          can_interact_with_elements: true
        };
        
      case 'gemini-test':
        return {
          page_type: 'gemini_ai_test',
          available_actions: ['test Gemini AI', 'send queries', 'view responses'],
          has_ai_capabilities: true
        };
        
      default:
        return {
          page_type: 'unknown',
          available_actions: ['navigate to other pages']
        };
    }
  }, []);
  
  // Debug logging
  console.log('🔧 [App] MCP Server URL from env:', process.env.REACT_APP_MCP_SERVER_URL);
  console.log('🔧 [App] Using MCP Server URL:', mcpServerUrl);
  
  // Use the new MCP automation system
  const { 
    executeCommand,
    isExecuting,
    isConnected,
    automationEnabled,
    toggleAutomation,
    serverUrl
  } = useMCPAutomation(mcpServerUrl);

  // Memoized handlers to prevent unnecessary re-renders
  const handleAddIntegration = useCallback(() => {
    setCurrentPage('available-integrations');
  }, []);

  const handleSelectIntegration = useCallback((integration) => {
    setSelectedIntegration(integration);
    setCurrentPage('add-integration');
  }, []);

  const handleBackToIntegrations = useCallback(() => {
    setCurrentPage('your-integrations');
  }, []);

  const handleCancelAddIntegration = useCallback(() => {
    setCurrentPage('available-integrations');
  }, []);

  const handleSubmitIntegration = useCallback(() => {
    setCurrentPage('your-integrations');
    setSelectedIntegration(null);
  }, []);

  const handleNavigate = useCallback((page) => {
    setCurrentPage(page);
    setSelectedIntegration(null);
  }, []);

  const handleSearchChange = useCallback((value) => {
    setSearchTerm(value);
  }, []);

  const handleToggleChat = useCallback(() => {
    setChatOpen(prev => !prev);
  }, []);

  const handleInputChange = useCallback((value) => {
    setInputMessage(value);
  }, []);

  const handleSendMessage = useCallback(async () => {
    if (!inputMessage.trim()) return;
    
    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      sender: 'user',
      content: inputMessage,
      timestamp: new Date(),
      type: 'text'
    };
    
    setChatMessages(prev => [...prev, userMessage]);
    
    try {
      console.log('🎯 [App] Processing command with MCP server:', inputMessage);
      
      // Show loading message
      const loadingMessage = {
        id: Date.now() + 1,
        sender: 'bot',
        content: `Processing: ${inputMessage}`,
        timestamp: new Date(),
        type: 'action'
      };
      setChatMessages(prev => [...prev, loadingMessage]);
      
      // Get comprehensive page context for MCP server
      const pageContext = {
        current_page: currentPage,
        available_pages: ['your-integrations', 'available-integrations', 'llm-demo', 'browser-automation', 'gemini-test'],
        search_term: searchTerm,
        selected_integration: selectedIntegration?.name || null,
        page_data: getCurrentPageData(currentPage, mockIntegrations, availableIntegrations, searchTerm, selectedIntegration),
        user_intent_context: {
          last_search: searchTerm,
          has_selected_integration: !!selectedIntegration,
          current_view: currentPage,
          available_integrations_count: availableIntegrations?.length || 0,
          user_integrations_count: mockIntegrations?.length || 0
        }
      };
      
      // Process command with MCP server (plan first, execute later)
      const result = await executeCommand(inputMessage, {
        executeActions: false, // First, just plan the actions
        autoConfirm: true,
        context: pageContext
      });
      
      console.log('🧠 [App] MCP result:', result);
      
      // Remove loading message
      setChatMessages(prev => prev.filter(msg => msg.id !== loadingMessage.id));
      
      // Handle the action based on MCP result
      if (result.success) {
        const actionSteps = result.llmResult?.actionSteps || [];
        
        // Show planned steps in chat for transparency
        if (actionSteps.length > 0) {
          const planMessage = {
            id: Date.now() + 2,
            sender: 'bot',
            content: `I've planned ${actionSteps.length} steps to ${inputMessage}:`,
            timestamp: new Date(),
            type: 'plan'
          };
          setChatMessages(prev => [...prev, planMessage]);

          // Show each planned step
          actionSteps.forEach((step, index) => {
            const stepMessage = {
              id: Date.now() + 10 + index,
              sender: 'bot',
              content: `${index + 1}. ${step.description}`,
              timestamp: new Date(),
              type: 'step',
              stepData: step
            };
            setChatMessages(prev => [...prev, stepMessage]);
          });

          // Show confidence and warnings
          const confidenceText = result.llmResult?.confidence 
            ? `Confidence: ${(result.llmResult.confidence * 100).toFixed(0)}%` 
            : '';
          
          const warningsText = result.llmResult?.warnings && result.llmResult.warnings.length > 0
            ? `⚠️ Warnings: ${result.llmResult.warnings.join(', ')}`
            : '';

          const summaryContent = [
            confidenceText,
            warningsText,
            'Executing these steps now...'
          ].filter(Boolean).join('\n');

          const summaryMessage = {
            id: Date.now() + 20,
            sender: 'bot',
            content: summaryContent,
            timestamp: new Date(),
            type: 'summary'
          };
          setChatMessages(prev => [...prev, summaryMessage]);

          // Wait a moment for user to see the plan, then execute
          setTimeout(async () => {
            try {
              // Show execution start message
              const executionStartMessage = {
                id: Date.now() + 25,
                sender: 'bot',
                content: '⚡ Starting execution...',
                timestamp: new Date(),
                type: 'action'
              };
              setChatMessages(prev => [...prev, executionStartMessage]);

              // Now execute the actions
              const executionResult = await executeCommand(inputMessage, {
                executeActions: true,
                autoConfirm: true,
                context: pageContext
              });

              // Remove execution start message
              setChatMessages(prev => prev.filter(msg => msg.id !== executionStartMessage.id));

              // Handle navigation if needed
              const navigationStep = actionSteps.find(step => 
                step.action_type === 'navigate' && 
                step.target && 
                ['your-integrations', 'available-integrations', 'llm-demo', 'browser-automation', 'gemini-test'].includes(step.target.replace('/', ''))
              );
              
              if (navigationStep) {
                const targetPage = navigationStep.target.replace('/', '');
                console.log('✅ [App] App navigation detected, page:', targetPage);
                setCurrentPage(targetPage);
              }

              // Show detailed execution results
              if (executionResult.success && executionResult.executionResult) {
                const { results } = executionResult.executionResult;
                const successCount = results?.filter(r => r.success).length || 0;
                const totalCount = results?.length || actionSteps.length;
                
                const resultMessage = {
                  id: Date.now() + 30,
                  sender: 'bot',
                  content: `✅ Execution completed! Successfully executed ${successCount}/${totalCount} steps.`,
                  timestamp: new Date(),
                  type: 'success'
                };
                setChatMessages(prev => [...prev, resultMessage]);

                // Show failed steps if any
                if (results && successCount < totalCount) {
                  const failedSteps = results.filter(r => !r.success);
                  failedSteps.forEach((failedStep, index) => {
                    const failMessage = {
                      id: Date.now() + 35 + index,
                      sender: 'bot',
                      content: `⚠️ Step failed: ${failedStep.error || 'Unknown error'}`,
                      timestamp: new Date(),
                      type: 'warning'
                    };
                    setChatMessages(prev => [...prev, failMessage]);
                  });
                }
              } else {
                const resultMessage = {
                  id: Date.now() + 30,
                  sender: 'bot',
                  content: executionResult.success 
                    ? `✅ Task completed successfully!`
                    : `❌ Execution failed: ${executionResult.error || 'Unknown error'}`,
                  timestamp: new Date(),
                  type: executionResult.success ? 'success' : 'error'
                };
                setChatMessages(prev => [...prev, resultMessage]);
              }

            } catch (executionError) {
              console.error('❌ [App] Error during execution:', executionError);
              
              // Remove execution start message if still present
              setChatMessages(prev => prev.filter(msg => msg.type !== 'action'));
              
              const errorMessage = {
                id: Date.now() + 30,
                sender: 'bot',
                content: `❌ Execution failed: ${executionError.message}`,
                timestamp: new Date(),
                type: 'error'
              };
              setChatMessages(prev => [...prev, errorMessage]);
            }
          }, 2000); // 2 second delay to show the plan

        } else {
          // No action steps generated
          const noStepsMessage = {
            id: Date.now() + 2,
            sender: 'bot',
            content: result.llmResult?.description || "I understand your request, but I couldn't generate specific action steps for it.",
            timestamp: new Date(),
            type: 'info'
          };
          setChatMessages(prev => [...prev, noStepsMessage]);
        }
        
      } else {
        throw new Error(result.error || 'Unknown error occurred');
      }
      
    } catch (error) {
      console.error('❌ [App] Error processing command:', error);
      
      // Remove loading message and add error
      setChatMessages(prev => prev.filter(msg => msg.type !== 'action'));
      
      const errorMessage = {
        id: Date.now() + 3,
        sender: 'bot',
        content: `Error: ${error.message}`,
        timestamp: new Date(),
        type: 'error'
      };
      
      setChatMessages(prev => [...prev, errorMessage]);
    }
    
    setInputMessage('');
  }, [inputMessage, executeCommand, setCurrentPage, currentPage, searchTerm, selectedIntegration]);

  // Enhanced quick actions with MCP server status
  const quickActions = useMemo(() => ({
    'What can you do?': () => {
      setChatMessages(prev => [...prev, {
        id: Date.now(),
        sender: 'bot',
        content: `🤖 I can help you with:\n\n• Navigate between pages (integrations, demos)\n• Search for specific integrations\n• Add new integrations\n• Fill out forms\n• Click buttons and links\n• Provide step-by-step execution plans\n\nJust tell me what you want to do in natural language!`,
        timestamp: new Date(),
        type: 'info'
      }]);
    },
    'Toggle Automation': () => {
      const newState = toggleAutomation();
      setChatMessages(prev => [...prev, {
        id: Date.now(),
        sender: 'bot',
        content: `Browser automation ${newState ? 'enabled' : 'disabled'}`,
        timestamp: new Date(),
        type: newState ? 'success' : 'warning'
      }]);
    },
    'Server Status': () => {
      setChatMessages(prev => [...prev, {
        id: Date.now(),
        sender: 'bot',
        content: `MCP Server: ${isConnected ? '🟢 Connected' : '🔴 Disconnected'} (${serverUrl})`,
        timestamp: new Date(),
        type: isConnected ? 'success' : 'error'
      }]);
    }
  }), [toggleAutomation]); // Removed isConnected and serverUrl to prevent unnecessary recreations

  // Memoized filtered integrations
  const filteredIntegrations = useMemo(() => 
    availableIntegrations.filter(integration =>
      integration.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      integration.description.toLowerCase().includes(searchTerm.toLowerCase())
    ), [searchTerm]
  );

  // Render current page based on state
  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'your-integrations':
        return (
          <YourIntegrationsPage 
            integrations={mockIntegrations}
            onAddIntegration={handleAddIntegration}
          />
        );
      
      case 'available-integrations':
        return (
          <AvailableIntegrationsPage 
            integrations={filteredIntegrations}
            searchTerm={searchTerm}
            onSearchChange={handleSearchChange}
            onSelectIntegration={handleSelectIntegration}
            onBack={handleBackToIntegrations}
          />
        );
      
      case 'add-integration':
        return selectedIntegration ? (
          <AddIntegrationPage 
            integration={selectedIntegration}
            onCancel={handleCancelAddIntegration}
            onSubmit={handleSubmitIntegration}
          />
        ) : null;

      case 'browser-automation':
        return <BrowserAutomationDemo />;

      case 'llm-demo':
        return <LLMCommandDemo />;

      case 'gemini-test':
        return <GeminiTestComponent />;
      
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <Header />
      
      <main className="p-6">
        {renderCurrentPage()}
      </main>

      <FloatingChatbot 
        isOpen={chatOpen}
        onToggle={handleToggleChat}
        messages={chatMessages}
        inputMessage={inputMessage}
        onInputChange={handleInputChange}
        onSendMessage={handleSendMessage}
        isConnected={isConnected}
        isLoading={false}
        isExecuting={isExecuting}
        automationEnabled={automationEnabled}
        quickActions={quickActions}
        serverInfo={{
          url: serverUrl,
          connected: isConnected
        }}
      />
    </div>
  );
};

export default App;