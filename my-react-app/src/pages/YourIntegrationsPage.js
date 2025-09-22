import React, { memo } from 'react';
import { Filter, Settings, Plus } from 'lucide-react';
import FilterDropdown from '../components/FilterDropdown';

const IntegrationRow = memo(({ integration }) => (
  <tr className="border-t border-gray-700 hover:bg-gray-750">
    <td className="px-6 py-4">
      <div className="flex items-center space-x-3">
        <span className="text-2xl">{integration.icon}</span>
        <span className="text-blue-400">{integration.name}</span>
      </div>
    </td>
    <td className="px-6 py-4">{integration.features}</td>
    <td className="px-6 py-4">
      <div className="flex items-center space-x-2">
        <span className="w-2 h-2 bg-green-500 rounded-full"></span>
        <span>{integration.status}</span>
      </div>
    </td>
    <td className="px-6 py-4 text-gray-400">{integration.uniqueConfig}</td>
    <td className="px-6 py-4">{integration.category}</td>
    <td className="px-6 py-4">{integration.eps}</td>
    <td className="px-6 py-4"></td>
    <td className="px-6 py-4 text-right">
      <button className="text-gray-400 hover:text-white">⋮</button>
    </td>
  </tr>
));

const PaginationButton = memo(({ number, isActive, onClick }) => (
  <button 
    onClick={onClick}
    className={`w-8 h-8 rounded ${isActive ? 'bg-blue-600' : 'bg-gray-700'}`}
  >
    {number}
  </button>
));

const YourIntegrationsPage = ({ integrations, onAddIntegration }) => (
  <div>
    <div className="mb-8">
      <h1 className="text-3xl font-semibold mb-2">Your Integrations</h1>
      <p className="text-gray-400">Configure your 3rd party and custom integrations to ingest data and configure response actions.</p>
    </div>

    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center space-x-4">
        <div className="relative">
          <Filter className="w-5 h-5 absolute left-3 top-3 text-gray-400" />
          <input
            type="text"
            placeholder="Filter"
            className="bg-gray-800 border border-gray-600 rounded-lg pl-10 pr-4 py-2 w-80"
          />
        </div>
        <FilterDropdown label="Features: All" />
        <FilterDropdown label="Status: All" />
        <FilterDropdown label="Category: All" />
      </div>
      <div className="flex items-center space-x-4">
        <button className="px-4 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 hover:bg-gray-600">
          Sync Integrations
        </button>
        <button 
          onClick={onAddIntegration}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center space-x-2"
        >
          <Plus className="w-4 h-4" />
          <span>Add New Integration</span>
        </button>
      </div>
    </div>

    <div className="text-right text-gray-400 mb-4">
      Showing 50 of 253 integrations
    </div>

    <div className="bg-gray-800 rounded-lg overflow-hidden">
      <table className="w-full">
        <thead className="bg-gray-700">
          <tr>
            <th className="text-left px-6 py-4 text-gray-300">Integration</th>
            <th className="text-left px-6 py-4 text-gray-300">Features</th>
            <th className="text-left px-6 py-4 text-gray-300">Unique Configuration</th>
            <th className="text-left px-6 py-4 text-gray-300">Category</th>
            <th className="text-left px-6 py-4 text-gray-300">EPS</th>
            <th className="text-left px-6 py-4 text-gray-300">Tags</th>
            <th className="text-right px-6 py-4 text-gray-300">
              <Settings className="w-4 h-4" />
            </th>
          </tr>
        </thead>
        <tbody>
          {integrations.map((integration) => (
            <IntegrationRow key={integration.id} integration={integration} />
          ))}
        </tbody>
      </table>
    </div>

    <div className="flex items-center justify-between mt-6">
      <div className="flex items-center space-x-2">
        <span className="text-gray-400">Rows</span>
        {[50, 100, 500, 1000].map(num => (
          <button key={num} className={`px-3 py-1 rounded ${num === 50 ? 'bg-blue-600' : 'bg-gray-700'}`}>
            {num}
          </button>
        ))}
      </div>
      <div className="flex items-center space-x-2">
        <span className="text-gray-400">1 - 50 of 253</span>
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

export default memo(YourIntegrationsPage); 