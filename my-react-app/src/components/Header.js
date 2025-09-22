import React from 'react';

const Header = ({ currentPage, onNavigate }) => {
  const handleNavigate = (page) => {
    if (onNavigate) {
      onNavigate(page);
    }
  };

  return (
    <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-blue-600 rounded-sm flex items-center justify-center">
              <span className="text-white font-bold text-sm">T</span>
            </div>
            <span className="text-xl font-semibold">TrelliX</span>
          </div>
          
          {/* Navigation Menu */}
          <nav className="flex items-center space-x-4">
            <button
              onClick={() => handleNavigate && handleNavigate('your-integrations')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                currentPage === 'your-integrations' 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              Integrations
            </button>
            <button
              onClick={() => handleNavigate && handleNavigate('browser-automation')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                currentPage === 'browser-automation' 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              Browser Automation
            </button>
            <button
              onClick={() => handleNavigate && handleNavigate('llm-demo')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                currentPage === 'llm-demo' 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              LLM Demo
            </button>
          </nav>
          
          {/* Breadcrumb */}
          <div className="flex items-center space-x-2 text-gray-400">
            <span>Helix</span>
            <span>→</span>
            <span className="text-blue-400">
              {currentPage === 'your-integrations' && 'Integration Hub'}
              {currentPage === 'available-integrations' && 'Available Integrations'}
              {currentPage === 'add-integration' && 'Add New Integration'}
              {currentPage === 'browser-automation' && 'Browser Automation System'}
              {currentPage === 'llm-demo' && 'LLM Command Demo'}
            </span>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <button className="p-2 hover:bg-gray-700 rounded">🌙</button>
          <button className="p-2 hover:bg-gray-700 rounded">↗</button>
        </div>
      </div>
    </header>
  );
};

export default Header; 