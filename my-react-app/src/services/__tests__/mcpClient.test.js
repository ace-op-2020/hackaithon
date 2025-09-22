/**
 * Test script to verify health check debouncing
 * This simulates rapid health check calls to ensure they're properly debounced
 */

import MCPClient from '../src/services/mcpClient.js';

// Mock fetch for testing
global.fetch = jest.fn();

describe('MCPClient Health Check Debouncing', () => {
  let mcpClient;

  beforeEach(() => {
    mcpClient = new MCPClient('http://localhost:8000');
    fetch.mockClear();
  });

  test('should debounce rapid health check calls', async () => {
    // Mock successful response
    fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ status: 'healthy' })
    });

    // Make 5 rapid health check calls
    const promises = [];
    for (let i = 0; i < 5; i++) {
      promises.push(mcpClient.checkHealth());
    }

    // Wait for all to complete
    const results = await Promise.all(promises);

    // Should only make one actual fetch call due to debouncing
    expect(fetch).toHaveBeenCalledTimes(1);
    
    // All should return the same result
    results.forEach(result => {
      expect(result.status).toBe('healthy');
    });
  });

  test('should allow health checks after debounce period', async () => {
    fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ status: 'healthy' })
    });

    // First health check
    await mcpClient.checkHealth();
    expect(fetch).toHaveBeenCalledTimes(1);

    // Simulate time passing (mock the lastHealthCheck)
    mcpClient.lastHealthCheck = Date.now() - 4000; // 4 seconds ago

    // Second health check should go through
    await mcpClient.checkHealth();
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  test('should cache recent health check results', async () => {
    fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ status: 'healthy' })
    });

    // First call
    const result1 = await mcpClient.checkHealth();
    expect(fetch).toHaveBeenCalledTimes(1);

    // Immediate second call should be cached
    const result2 = await mcpClient.checkHealth();
    expect(fetch).toHaveBeenCalledTimes(1); // Still just one call
    expect(result2.status).toBe('cached');
  });
});
