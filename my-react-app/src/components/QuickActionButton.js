import React, { memo } from 'react';

const QuickActionButton = ({ text, onClick, disabled = false }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`text-xs px-2 py-1 rounded border transition-colors ${
      disabled
        ? 'bg-gray-800 text-gray-500 border-gray-700 cursor-not-allowed'
        : 'bg-gray-700 hover:bg-gray-600 text-gray-300 border-gray-600 hover:border-gray-500'
    }`}
  >
    {text}
  </button>
);

export default memo(QuickActionButton); 