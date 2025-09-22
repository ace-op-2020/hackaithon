/**
 * Popup Script for TrelliX Browser Extension
 * Handles the extension popup UI and interactions
 */

document.addEventListener('DOMContentLoaded', async () => {
  const statusIndicator = document.getElementById('statusIndicator');
  const statusText = document.getElementById('statusText');
  const openAppBtn = document.getElementById('openApp');
  const toggleExtensionBtn = document.getElementById('toggleExtension');

  // Check if TrelliX app is running
  await checkAppConnection();

  // Set up event listeners
  openAppBtn.addEventListener('click', openTrelliXApp);
  toggleExtensionBtn.addEventListener('click', toggleExtension);

  /**
   * Check connection to TrelliX app
   */
  async function checkAppConnection() {
    try {
      // Try to ping the TrelliX app
      const response = await fetch('http://localhost:3000', {
        method: 'HEAD',
        mode: 'no-cors'
      });

      updateStatus(true, 'Connected to TrelliX App');
    } catch (error) {
      updateStatus(false, 'TrelliX App not running');
    }
  }

  /**
   * Update status display
   */
  function updateStatus(isConnected, message) {
    statusIndicator.className = `status-indicator ${isConnected ? 'active' : 'inactive'}`;
    statusText.textContent = message;
  }

  /**
   * Open TrelliX app
   */
  async function openTrelliXApp() {
    try {
      await chrome.tabs.create({
        url: 'http://localhost:3000',
        active: true
      });
      window.close();
    } catch (error) {
      console.error('Failed to open TrelliX app:', error);
    }
  }

  /**
   * Toggle extension enabled/disabled state
   */
  async function toggleExtension() {
    try {
      const { extensionEnabled = true } = await chrome.storage.sync.get('extensionEnabled');
      const newState = !extensionEnabled;
      
      await chrome.storage.sync.set({ extensionEnabled: newState });
      
      toggleExtensionBtn.textContent = newState ? 'Disable Extension' : 'Enable Extension';
      updateStatus(newState, newState ? 'Extension Enabled' : 'Extension Disabled');
      
    } catch (error) {
      console.error('Failed to toggle extension:', error);
    }
  }

  /**
   * Get current tab info
   */
  async function getCurrentTabInfo() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      return tab;
    } catch (error) {
      console.error('Failed to get current tab:', error);
      return null;
    }
  }

  // Initialize extension state
  const { extensionEnabled = true } = await chrome.storage.sync.get('extensionEnabled');
  toggleExtensionBtn.textContent = extensionEnabled ? 'Disable Extension' : 'Enable Extension';
});
