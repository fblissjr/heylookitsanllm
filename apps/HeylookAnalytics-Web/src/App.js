import React, { useState } from 'react';
import axios from 'axios';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { Activity, Database, Zap, AlertCircle, RefreshCw, Play, Save, Send, RotateCcw, Eye, Layers } from 'lucide-react';
import RequestInspector from './components/RequestInspector';
import EvaluationTool from './components/EvaluationTool';
import PerformanceProfiler from './components/PerformanceProfiler';
import BatchProcessor from './components/BatchProcessor';
import './App.css';

// Configure axios defaults
axios.defaults.headers.common['Content-Type'] = 'application/json';

const queryClient = new QueryClient();
const API_BASE = 'http://localhost:8080';

// Main App Component
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="app">
        <header className="app-header">
          <h1>HeylookAnalytics</h1>
          <p>Real-time monitoring for heylookitsanllm</p>
        </header>
        <Dashboard />
      </div>
    </QueryClientProvider>
  );
}

// Dashboard Component
function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');

  // Check connection
  const { data: isConnected } = useQuery({
    queryKey: ['connection'],
    queryFn: async () => {
      try {
        await axios.get(`${API_BASE}/v1/models`);
        return true;
      } catch {
        return false;
      }
    },
    refetchInterval: 30000,
  });

  // Fetch models
  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/v1/models`);
      return response.data.data;
    },
    enabled: isConnected,
  });

  return (
    <div className="dashboard">
      <nav className="nav-tabs">
        <button 
          className={activeTab === 'overview' ? 'active' : ''} 
          onClick={() => setActiveTab('overview')}
        >
          <Activity size={16} /> Overview
        </button>
        <button 
          className={activeTab === 'playground' ? 'active' : ''} 
          onClick={() => setActiveTab('playground')}
        >
          <Play size={16} /> Playground
        </button>
        <button 
          className={activeTab === 'compare' ? 'active' : ''} 
          onClick={() => setActiveTab('compare')}
        >
          <Zap size={16} /> Compare
        </button>
        <button 
          className={activeTab === 'conversation' ? 'active' : ''} 
          onClick={() => setActiveTab('conversation')}
        >
          <RefreshCw size={16} /> Conversation
        </button>
        <button 
          className={activeTab === 'query' ? 'active' : ''} 
          onClick={() => setActiveTab('query')}
        >
          <Database size={16} /> Query
        </button>
        <button 
          className={activeTab === 'inspector' ? 'active' : ''} 
          onClick={() => setActiveTab('inspector')}
        >
          <Eye size={16} /> Inspector
        </button>
        <button 
          className={activeTab === 'models' ? 'active' : ''} 
          onClick={() => setActiveTab('models')}
        >
          <Zap size={16} /> Models
        </button>
        <button 
          className={activeTab === 'batch' ? 'active' : ''} 
          onClick={() => setActiveTab('batch')}
        >
          <Layers size={16} /> Batch
        </button>
      </nav>

      <div className="tab-content">
        {!isConnected && (
          <div className="alert alert-error">
            <AlertCircle size={16} />
            Not connected to server. Make sure heylookitsanllm is running on {API_BASE}
          </div>
        )}

        {activeTab === 'overview' && <PerformanceProfiler />}
        {activeTab === 'playground' && <PlaygroundTab models={models} />}
        {activeTab === 'compare' && <CompareTab models={models} />}
        {activeTab === 'conversation' && <ConversationTab models={models} />}
        {activeTab === 'query' && <QueryTab />}
        {activeTab === 'inspector' && <RequestInspector />}
        {activeTab === 'models' && <ModelsTab models={models} />}
        {activeTab === 'batch' && <BatchProcessor />}
      </div>
    </div>
  );
}

// Query Tab
function QueryTab() {
  const [query, setQuery] = useState('SELECT * FROM request_logs LIMIT 10');
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const executeQuery = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/v1/data/query`, {
        query,
        limit: 1000
      });
      setResults(response.data);
    } catch (error) {
      if (error.response?.status === 503) {
        setResults({ 
          error: "Analytics not enabled", 
          message: "To enable analytics:\n1. Create a .env file in the server directory\n2. Add: HEYLOOK_ANALYTICS_ENABLED=true\n3. Add: HEYLOOK_ANALYTICS_STORAGE_LEVEL=basic\n4. Restart the server"
        });
      } else {
        setResults({ error: error.response?.data?.detail || error.message });
      }
    }
    setIsLoading(false);
  };

  return (
    <div className="query-tab">
      <div className="query-editor">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter SQL query..."
          rows={6}
        />
        <div className="query-actions">
          <button onClick={executeQuery} disabled={isLoading}>
            {isLoading ? <RefreshCw size={16} className="spin" /> : <Play size={16} />}
            Execute
          </button>
          <button className="secondary">
            <Save size={16} /> Save Query
          </button>
        </div>
      </div>

      {results && (
        <div className="query-results">
          {results.error ? (
            <div className="alert alert-error">
              <div>{results.error}</div>
              {results.message && (
                <pre style={{ marginTop: '10px', whiteSpace: 'pre-wrap' }}>{results.message}</pre>
              )}
            </div>
          ) : (
            <>
              <p>{results.row_count} rows returned</p>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      {results.columns.map(col => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.data.slice(0, 100).map((row, i) => (
                      <tr key={i}>
                        {row.map((cell, j) => (
                          <td key={j}>{JSON.stringify(cell)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// Models Tab
function ModelsTab({ models }) {
  return (
    <div className="models-tab">
      <h3>Available Models ({models?.length || 0})</h3>
      <div className="models-grid">
        {models?.map(model => (
          <div key={model.id} className="model-card">
            <h4>{model.id}</h4>
            <p className="model-meta">Created: {new Date(model.created * 1000).toLocaleDateString()}</p>
            <div className="model-tags">
              {model.id.includes('vision') && <span className="tag">Vision</span>}
              {model.id.includes('mlx') && <span className="tag">MLX</span>}
              {model.id.includes('gguf') && <span className="tag">GGUF</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Playground Tab - Evaluation and Testing
function PlaygroundTab({ models }) {
  return <EvaluationTool />;
}

// Compare Tab - Model comparison
function CompareTab({ models }) {
  const [prompt, setPrompt] = useState('');
  const [selectedModels, setSelectedModels] = useState([]);
  const [results, setResults] = useState({});
  const [isComparing, setIsComparing] = useState(false);

  const toggleModel = (modelId) => {
    setSelectedModels(prev => 
      prev.includes(modelId) 
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  const runComparison = async () => {
    if (!prompt || selectedModels.length === 0) return;
    
    setIsComparing(true);
    setResults({});

    // Run all requests in parallel
    const promises = selectedModels.map(async (modelId) => {
      const startTime = Date.now();
      
      try {
        const response = await axios.post(`${API_BASE}/v1/chat/completions`, {
          model: modelId,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.7,
          max_tokens: 1000,
        });

        const endTime = Date.now();
        
        return {
          modelId,
          response: response.data.choices[0].message.content,
          responseTime: endTime - startTime,
          usage: response.data.usage,
          success: true,
        };
      } catch (error) {
        return {
          modelId,
          error: error.message,
          success: false,
        };
      }
    });

    const allResults = await Promise.all(promises);
    
    const resultsMap = {};
    allResults.forEach(result => {
      resultsMap[result.modelId] = result;
    });
    
    setResults(resultsMap);
    setIsComparing(false);
  };

  return (
    <div className="compare-tab">
      <div className="compare-prompt">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a prompt to compare across models..."
          rows={4}
        />
      </div>

      <div className="model-selector">
        <h4>Select Models to Compare</h4>
        <div className="model-checkboxes">
          {models?.map(model => (
            <label key={model.id} className="model-checkbox">
              <input
                type="checkbox"
                checked={selectedModels.includes(model.id)}
                onChange={() => toggleModel(model.id)}
              />
              <span>{model.id}</span>
            </label>
          ))}
        </div>
      </div>

      <button 
        onClick={runComparison} 
        disabled={isComparing || !prompt || selectedModels.length === 0}
        className="compare-button"
      >
        {isComparing ? <RefreshCw size={16} className="spin" /> : <Zap size={16} />}
        {isComparing ? 'Comparing...' : 'Compare Models'}
      </button>

      {Object.keys(results).length > 0 && (
        <div className="comparison-results">
          {selectedModels.map(modelId => {
            const result = results[modelId];
            if (!result) return null;
            
            return (
              <div key={modelId} className="comparison-card">
                <h4>{modelId}</h4>
                
                {result.success ? (
                  <>
                    <div className="comparison-metrics">
                      <span className="metric">Time: {result.responseTime}ms</span>
                      <span className="metric">Tokens: {result.usage?.total_tokens || 0}</span>
                      <span className="metric">TPS: {result.usage?.completion_tokens && result.responseTime ? 
                        ((result.usage.completion_tokens / result.responseTime) * 1000).toFixed(1) : 'N/A'}</span>
                    </div>
                    <div className="comparison-response">
                      <pre>{result.response}</pre>
                    </div>
                  </>
                ) : (
                  <div className="error-message">Error: {result.error}</div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// Conversation Tab - Multi-turn conversation testing
function ConversationTab({ models }) {
  const [selectedModel, setSelectedModel] = useState(models?.[0]?.id || '');
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    if (!inputMessage || !selectedModel) return;
    
    const userMessage = { role: 'user', content: inputMessage };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/v1/chat/completions`, {
        model: selectedModel,
        messages: newMessages,
        temperature: 0.7,
        max_tokens: 1000,
      });

      const assistantMessage = response.data.choices[0].message;
      setMessages([...newMessages, assistantMessage]);
    } catch (error) {
      const errorMessage = { 
        role: 'system', 
        content: `Error: ${error.response?.data?.detail || error.message}` 
      };
      setMessages([...newMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearConversation = () => {
    setMessages([]);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="conversation-tab">
      <div className="conversation-controls">
        <select 
          value={selectedModel} 
          onChange={(e) => setSelectedModel(e.target.value)}
        >
          {models?.map(model => (
            <option key={model.id} value={model.id}>{model.id}</option>
          ))}
        </select>
        
        <button onClick={clearConversation} className="clear-button">
          <RotateCcw size={16} /> Clear
        </button>
      </div>

      <div className="conversation-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            Start a conversation by typing a message below
          </div>
        )}
        
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <div className="message-role">{message.role}</div>
            <div className="message-content">{message.content}</div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message assistant loading">
            <div className="message-role">assistant</div>
            <div className="message-content">
              <RefreshCw size={16} className="spin" /> Thinking...
            </div>
          </div>
        )}
      </div>

      <div className="conversation-input">
        <textarea
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
          rows={3}
          disabled={isLoading}
        />
        <button 
          onClick={sendMessage} 
          disabled={isLoading || !inputMessage || !selectedModel}
        >
          <Send size={16} /> Send
        </button>
      </div>
    </div>
  );
}

export default App;