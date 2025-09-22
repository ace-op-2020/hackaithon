#!/usr/bin/env node

/**
 * Vertex AI Gemini Test with Service Account Credentials
 * 
 * This script tests Gemini using your service account credentials
 */

import { readFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { config } from 'dotenv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables
config();

console.log('Vertex AI Gemini Test with Service Account');
console.log('==========================================');

async function testVertexAIGemini() {
  try {
    // 1. Check service account file
    console.log('\n1. SERVICE ACCOUNT CREDENTIALS CHECK');
    console.log('-'.repeat(40));
    
    const serviceAccountPath = join(__dirname, 'svc-hackathon-prod15-f4c7ef8a79c2.json');
    
    if (!existsSync(serviceAccountPath)) {
      console.log('ERROR: Service account file not found');
      return;
    }
    
    const serviceAccount = JSON.parse(readFileSync(serviceAccountPath, 'utf8'));
    console.log('✓ Service account file loaded');
    console.log('✓ Project ID:', serviceAccount.project_id);
    console.log('✓ Client Email:', serviceAccount.client_email);
    console.log('✓ Private Key:', serviceAccount.private_key ? 'Present' : 'Missing');
    
    // 2. Test Google Auth Library
    console.log('\n2. AUTHENTICATION TEST');
    console.log('-'.repeat(40));
    
    try {
      const { GoogleAuth } = await import('google-auth-library');
      console.log('✓ Google Auth Library imported');
      
      // Create auth client with service account
      const auth = new GoogleAuth({
        keyFile: serviceAccountPath,
        scopes: ['https://www.googleapis.com/auth/cloud-platform']
      });
      
      console.log('✓ Auth client created');
      
      // Get access token
      const authClient = await auth.getClient();
      console.log('✓ Auth client authenticated');
      
      const accessToken = await authClient.getAccessToken();
      console.log('✓ Access token obtained:', accessToken.token ? 'Success' : 'Failed');
      
      // 3. Test Vertex AI API call
      console.log('\n3. VERTEX AI GEMINI API TEST');
      console.log('-'.repeat(40));
      
      const projectId = serviceAccount.project_id;
      const location = 'us-central1';
      const model = 'gemini-2.5-flash-lite';
      
      // Prepare the API request
      const endpoint = `https://${location}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${location}/publishers/google/models/${model}:generateContent`;
      
      const requestBody = {
        contents: [{
          role: 'user',
          parts: [{
            text: 'Parse this browser command and return JSON: "click the login button". Return format: {"action": "click", "target": "login button", "confidence": "high"}'
          }]
        }],
        generationConfig: {
          temperature: 0.1,
          maxOutputTokens: 256,
          topP: 0.8,
          topK: 40
        }
      };
      
      console.log('✓ API endpoint:', endpoint);
      console.log('✓ Request body prepared');
      
      // Make the API call
      console.log('\n📡 Making API call to Gemini...');
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken.token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      
      console.log('✓ API call completed');
      console.log('✓ Status:', response.status, response.statusText);
      
      if (response.ok) {
        const data = await response.json();
        console.log('\n🎉 SUCCESS: Gemini API is working!');
        console.log('Response:', JSON.stringify(data, null, 2));
        
        if (data.candidates && data.candidates[0] && data.candidates[0].content) {
          const generatedText = data.candidates[0].content.parts[0].text;
          console.log('\n📝 Generated text:', generatedText);
          
          // Try to parse as JSON
          try {
            const parsedResult = JSON.parse(generatedText);
            console.log('✓ Parsed JSON result:', parsedResult);
          } catch (e) {
            console.log('⚠ Generated text is not valid JSON, but API is working');
          }
        }
        
      } else {
        const errorData = await response.text();
        console.log('❌ API call failed');
        console.log('Error response:', errorData);
      }
      
    } catch (authError) {
      console.log('❌ Authentication failed:', authError.message);
      console.log('Full error:', authError);
    }
    
  } catch (error) {
    console.log('❌ Test failed:', error.message);
    console.log('Stack:', error.stack);
  }
}

// 4. Test your existing service integration
async function testExistingService() {
  console.log('\n4. EXISTING SERVICE INTEGRATION TEST');
  console.log('-'.repeat(40));
  
  try {
    const servicePath = join(__dirname, 'src/services/googleCloudService.js');
    
    if (existsSync(servicePath)) {
      console.log('✓ Service file exists');
      
      // Import the service
      const serviceModule = await import(`file:///${servicePath.replace(/\\/g, '/')}`);
      const service = serviceModule.default;
      
      if (service && typeof service.parseBrowserCommand === 'function') {
        console.log('✓ Service imported successfully');
        
        // Test the service with your credentials
        console.log('\n📝 Testing parseBrowserCommand...');
        const result = await service.parseBrowserCommand('click the submit button');
        
        console.log('✓ Command processed');
        console.log('Result:', JSON.stringify(result, null, 2));
        
        if (result.source === 'gemini') {
          console.log('🎉 Your service is using Gemini API!');
        } else {
          console.log('⚠ Service is using fallback parser (may need credential setup)');
        }
        
      } else {
        console.log('❌ Service not properly exported');
      }
    } else {
      console.log('❌ Service file not found');
    }
    
  } catch (error) {
    console.log('❌ Service test failed:', error.message);
  }
}

// Run all tests
async function runAllTests() {
  await testVertexAIGemini();
  await testExistingService();
  
  console.log('\n' + '='.repeat(50));
  console.log('TEST SUMMARY');
  console.log('='.repeat(50));
  console.log('✓ If you see "SUCCESS: Gemini API is working!" above,');
  console.log('  your credentials are valid and Gemini is accessible.');
  console.log('');
  console.log('⚠ If you see authentication or API errors:');
  console.log('  1. Check that the service account has Vertex AI access');
  console.log('  2. Ensure the project has Vertex AI API enabled');
  console.log('  3. Verify the service account permissions');
  console.log('');
  console.log('🚀 Next: Update your .env to use these credentials for the React app');
}

runAllTests().catch(console.error);
