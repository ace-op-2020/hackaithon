import React, { useState, memo, useMemo } from 'react';
import { Plus, ChevronDown } from 'lucide-react';
import { availableIntegrations } from '../data/mockData';

const AddIntegrationPage = ({ integration, onCancel, onSubmit }) => {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    tags: '',
    ingestionEnabled: false
  });

  const integrationDetails = useMemo(() => 
    availableIntegrations.find(i => i.id === integration), 
    [integration]
  );

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const isFormValid = formData.name.trim().length > 0;

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-semibold">Add New Integrations</h1>
        <div className="flex space-x-4">
          <button 
            onClick={onCancel}
            className="px-4 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 hover:bg-gray-600"
          >
            Cancel
          </button>
          <button 
            onClick={onSubmit}
            disabled={!isFormValid}
            className={`px-4 py-2 rounded-lg transition-colors ${
              isFormValid 
                ? 'bg-blue-600 text-white hover:bg-blue-700' 
                : 'bg-gray-600 text-gray-400 cursor-not-allowed'
            }`}
          >
            Add and Verify Integration
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-6">Details</h2>
            
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Integration Name <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  placeholder="Integration Name"
                  value={formData.name}
                  onChange={(e) => handleInputChange('name', e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Description</label>
                <textarea
                  placeholder="Description"
                  value={formData.description}
                  onChange={(e) => handleInputChange('description', e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 h-24 focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Tags</label>
                <div className="flex">
                  <input
                    type="text"
                    placeholder="Tags"
                    value={formData.tags}
                    onChange={(e) => handleInputChange('tags', e.target.value)}
                    className="flex-1 bg-gray-700 border border-gray-600 rounded-l-lg px-3 py-2 focus:outline-none focus:border-blue-500"
                  />
                  <button className="bg-gray-600 border border-gray-600 rounded-r-lg px-3 py-2 hover:bg-gray-500">
                    <Plus className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            <div className="mt-8">
              <h3 className="text-lg font-semibold mb-4">Feature Settings</h3>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <input
                    type="checkbox"
                    checked={formData.ingestionEnabled}
                    onChange={(e) => handleInputChange('ingestionEnabled', e.target.checked)}
                    className="mt-1"
                  />
                  <div>
                    <div className="font-medium">Ingestion</div>
                    <div className="text-gray-400 text-sm">
                      This Helix integration will forward any files found in a given (AWS Access Verified) S3 bucket to Helix.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-1">
          {integrationDetails && (
            <div className="bg-gray-800 rounded-lg p-6">
              <div className="flex items-center space-x-3 mb-4">
                <span className="text-3xl">{integrationDetails.icon}</span>
                <h3 className="text-xl font-semibold">{integrationDetails.name}</h3>
              </div>
              <p className="text-gray-400 mb-4">{integrationDetails.description}</p>
              <div className="bg-gray-700 text-gray-300 text-xs px-2 py-1 rounded inline-block mb-6">
                {integrationDetails.category}
              </div>
              
              <div className="border-t border-gray-700 pt-4">
                <button className="flex items-center space-x-2 text-blue-400 w-full hover:text-blue-300">
                  <ChevronDown className="w-4 h-4" />
                  <span>Choose Bucket</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default memo(AddIntegrationPage); 