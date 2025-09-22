import { useCallback } from 'react';

export function useUIActions({ setCurrentPage, setSearchTerm, setSelectedIntegration }) {
  return useCallback(async (step) => {
    switch (step) {
      case 'navigate_to_available':
        setCurrentPage('available-integrations');
        break;
      case 'search_integration':
        setSearchTerm('amazon-verified-access');
        break;
      case 'select_integration':
        setSelectedIntegration('amazon-verified-access');
        setCurrentPage('add-integration');
        break;
      case 'fill_form':
        // Optionally auto-fill form data here
        break;
      default:
        break;
    }
  }, [setCurrentPage, setSearchTerm, setSelectedIntegration]);
} 