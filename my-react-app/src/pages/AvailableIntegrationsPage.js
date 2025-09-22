import React, { memo, useMemo } from 'react';
import { Filter } from 'lucide-react';
import FilterDropdown from '../components/FilterDropdown';

const IntegrationCard = memo(({ integration, onSelect }) => (
  <div 
    className="bg-gray-800 rounded-lg p-6 hover:bg-gray-750 cursor-pointer border border-gray-700 transition-colors"
    onClick={() => onSelect(integration.id)}
  >
    <div className="flex items-center justify-between mb-4">
      <h3 className="font-semibold text-white">{integration.name}</h3>
      <span className="text-2xl">{integration.icon}</span>
    </div>
    <p className="text-gray-400 text-sm mb-4">{integration.description}</p>
    <div className="bg-gray-700 text-gray-300 text-xs px-2 py-1 rounded inline-block">
      {integration.category}
    </div>
  </div>
));

const PaginationButton = memo(({ number, isActive, onClick }) => (
  <button 
    onClick={onClick}
    className={`w-8 h-8 rounded ${isActive ? 'bg-blue-600' : 'bg-gray-700'}`}
  >
    {number}
  </button>
));

const AvailableIntegrationsPage = ({ 
  integrations, 
  searchTerm, 
  onSearchChange, 
  onSelectIntegration, 
  onBack 
}) => {
  const filteredIntegrations = useMemo(() => 
    integrations.filter(integration =>
      integration.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      integration.description.toLowerCase().includes(searchTerm.toLowerCase())
    ), [integrations, searchTerm]
  );

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-semibold mb-2">Available Integrations</h1>
          <p className="text-gray-400">Extend the capabilities of your SecOps environment, including monitoring, analysis, workflow, triage, remediation and more.</p>
        </div>
        <button 
          onClick={onBack}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Your Integrations
        </button>
      </div>

      <div className="flex items-center space-x-4 mb-6">
        <div className="relative">
          <Filter className="w-5 h-5 absolute left-3 top-3 text-gray-400" />
          <input
            type="text"
            placeholder="Filter"
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            className="bg-gray-800 border border-gray-600 rounded-lg pl-10 pr-4 py-2 w-80"
          />
        </div>
        <FilterDropdown label="Features: All" />
        <FilterDropdown label="Category: All" />
      </div>

      <div className="text-right text-gray-400 mb-4">
        Showing {filteredIntegrations.length} of {integrations.length} integrations
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {filteredIntegrations.map((integration) => (
          <IntegrationCard 
            key={integration.id} 
            integration={integration}
            onSelect={onSelectIntegration}
          />
        ))}
      </div>

      <div className="flex items-center justify-center mt-8">
        <div className="flex items-center space-x-2">
          <span className="text-gray-400">1 - {filteredIntegrations.length} of {integrations.length}</span>
          <div className="flex space-x-1">
            {[1, 2, 3, 4, 5].map(num => (
              <PaginationButton 
                key={num} 
                number={num} 
                isActive={num === 1}
                onClick={() => {}}
              />
            ))}
            <button className="w-8 h-8 bg-gray-700 rounded">›</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default memo(AvailableIntegrationsPage); 