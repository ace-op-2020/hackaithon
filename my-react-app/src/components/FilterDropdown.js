import React from 'react';
import { ChevronDown } from 'lucide-react';

const FilterDropdown = ({ label, options = [], onSelect }) => {
  return (
    <div className="relative">
      <button className="flex items-center space-x-2 bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 hover:bg-gray-700">
        <span>{label}</span>
        <ChevronDown className="w-4 h-4" />
      </button>
    </div>
  );
};

export default FilterDropdown; 