/**
 * Background Script for TrelliX Browser Extension
 * Handles extension lifecycle and communication
 */

// Extension installation
chrome.runtime.onInstalled.addListener((details) => {
  console.log('TrelliX Browser Extension installed:', details);
  
  // Set up context menus
  chrome.contextMenus.create({
    id: 'trellix-automation',
    title: 'TrelliX Browser Automation',
    contexts: ['page', 'selection']
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'trellix-automation') {
    // Open TrelliX app in new tab
    chrome.tabs.create({
      url: 'http://localhost:3000'
    });
  }
});

// Handle messages from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Background script received message:', message);
  
  switch (message.type) {
    case 'GET_TAB_INFO':
      sendResponse({
        tabId: sender.tab.id,
        url: sender.tab.url,
        title: sender.tab.title
      });
      break;
      
    case 'TAKE_SCREENSHOT':
      handleScreenshot(sender.tab.id, sendResponse);
      return true; // Indicates async response
      
    default:
      sendResponse({ error: 'Unknown message type' });
  }
});

/**
 * Handle screenshot requests
 */
async function handleScreenshot(tabId, sendResponse) {
  try {
    const screenshot = await chrome.tabs.captureVisibleTab(null, {
      format: 'png',
      quality: 80
    });
    
    sendResponse({
      success: true,
      imageData: screenshot
    });
  } catch (error) {
    sendResponse({
      success: false,
      error: error.message
    });
  }
}

// Handle extension errors
chrome.runtime.onSuspend.addListener(() => {
  console.log('TrelliX Browser Extension suspended');
});

console.log('TrelliX Browser Extension background script loaded');
