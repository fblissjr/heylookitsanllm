import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Play, Download, AlertCircle, CheckCircle, XCircle, RefreshCw, FileJson, FilePlus } from 'lucide-react';
import './BatchProcessor.css';

const API_BASE = 'http://localhost:8080';

function BatchProcessor() {
  const [prompts, setPrompts] = useState([]);
  const [batchId, setBatchId] = useState(null); // eslint-disable-line no-unused-vars
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef();
  
  // Batch configuration
  const [defaultModel, setDefaultModel] = useState('qwen2.5-coder-1.5b-instruct-4bit');
  const [defaultTemperature, setDefaultTemperature] = useState(0.7);
  const [defaultMaxTokens, setDefaultMaxTokens] = useState(1000);
  const [parallelism, setParallelism] = useState(3);
  const [retryFailed, setRetryFailed] = useState(true);
  const [maxRetries, setMaxRetries] = useState(2);
  
  // Models list (should be fetched from API)
  const [models, setModels] = useState([]);

  React.useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_BASE}/v1/models`);
      setModels(response.data.data || []);
    } catch (err) {
      console.error('Error fetching models:', err);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target.result;
        let data;
        
        if (file.name.endsWith('.json')) {
          data = JSON.parse(content);
          if (data.prompts && Array.isArray(data.prompts)) {
            setPrompts(data.prompts);
          } else if (Array.isArray(data)) {
            // If it's just an array of prompts
            setPrompts(data.map((p, i) => ({
              id: p.id || `prompt_${i}`,
              content: p.content || p.prompt || p,
              ...p
            })));
          }
        } else if (file.name.endsWith('.csv')) {
          // Parse CSV
          const lines = content.split('\n').filter(line => line.trim());
          const headers = lines[0].split(',').map(h => h.trim());
          const promptsData = [];
          
          for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            const prompt = {};
            headers.forEach((header, idx) => {
              prompt[header] = values[idx];
            });
            promptsData.push({
              id: prompt.id || `prompt_${i-1}`,
              content: prompt.content || prompt.prompt,
              ...prompt
            });
          }
          setPrompts(promptsData);
        }
        
        setError(null);
      } catch (err) {
        setError(`Failed to parse file: ${err.message}`);
      }
    };
    
    reader.readAsText(file);
  };

  const addPrompt = () => {
    setPrompts([...prompts, {
      id: `prompt_${prompts.length}`,
      content: '',
      model: null,
      temperature: null,
      max_tokens: null,
      metadata: {}
    }]);
  };

  const updatePrompt = (index, field, value) => {
    const updated = [...prompts];
    updated[index][field] = value;
    setPrompts(updated);
  };

  const removePrompt = (index) => {
    setPrompts(prompts.filter((_, i) => i !== index));
  };

  const runBatch = async () => {
    if (prompts.length === 0) {
      setError('No prompts to process');
      return;
    }
    
    setLoading(true);
    setError(null);
    setBatchId(null);
    setResults(null);
    setStatus(null);
    
    try {
      // Filter out empty prompts
      const validPrompts = prompts.filter(p => p.content && p.content.trim());
      
      const response = await axios.post(`${API_BASE}/v1/batch/process`, {
        prompts: validPrompts,
        defaults: {
          model: defaultModel,
          temperature: defaultTemperature,
          max_tokens: defaultMaxTokens
        },
        batch_config: {
          parallelism: parallelism,
          retry_failed: retryFailed,
          max_retries: maxRetries
        }
      });
      
      setBatchId(response.data.batch_id);
      
      // Start polling for status
      pollBatchStatus(response.data.batch_id);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start batch processing');
      setLoading(false);
    }
  };

  const pollBatchStatus = async (batchId) => {
    const poll = async () => {
      try {
        const response = await axios.get(`${API_BASE}/v1/batch/${batchId}`);
        setStatus(response.data);
        
        if (response.data.status === 'completed' || response.data.status === 'failed') {
          setResults(response.data);
          setLoading(false);
        } else {
          setTimeout(poll, 1000); // Poll every second
        }
      } catch (err) {
        setError('Failed to fetch batch status');
        setLoading(false);
      }
    };
    
    poll();
  };

  const downloadResults = () => {
    if (!results) return;
    
    const exportData = {
      batch_id: results.batch_id,
      created_at: results.created_at,
      status: results.status,
      summary: {
        total: results.progress.total,
        completed: results.progress.completed,
        failed: results.progress.failed
      },
      results: results.results,
      errors: results.errors
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch_results_${results.batch_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadTemplate = () => {
    const template = {
      prompts: [
        {
          id: "example_1",
          content: "Explain quantum computing in simple terms",
          model: "qwen2.5-coder-1.5b-instruct-4bit",
          temperature: 0.7,
          max_tokens: 500,
          metadata: { category: "science" }
        },
        {
          id: "example_2",
          content: "Write a Python function to calculate fibonacci numbers",
          metadata: { category: "coding" }
        }
      ],
      defaults: {
        model: "qwen2.5-coder-1.5b-instruct-4bit",
        temperature: 0.7,
        max_tokens: 1000
      }
    };
    
    const blob = new Blob([JSON.stringify(template, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'batch_template.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="batch-processor">
      <div className="batch-header">
        <h2>Batch Processing</h2>
        <div className="batch-actions">
          <button onClick={downloadTemplate} className="template-button">
            <FileJson size={16} />
            Download Template
          </button>
          <button onClick={() => fileInputRef.current?.click()} className="upload-button">
            <Upload size={16} />
            Upload File
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,.csv"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
        </div>
      </div>

      {error && (
        <div className="error-message">
          <AlertCircle size={20} />
          {error}
        </div>
      )}

      <div className="batch-config">
        <h3>Batch Configuration</h3>
        <div className="config-grid">
          <div className="config-item">
            <label>Default Model</label>
            <select value={defaultModel} onChange={(e) => setDefaultModel(e.target.value)}>
              {models.map(model => (
                <option key={model.id} value={model.id}>{model.id}</option>
              ))}
            </select>
          </div>
          
          <div className="config-item">
            <label>Temperature: {defaultTemperature}</label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={defaultTemperature}
              onChange={(e) => setDefaultTemperature(Number(e.target.value))}
            />
          </div>
          
          <div className="config-item">
            <label>Max Tokens: {defaultMaxTokens}</label>
            <input
              type="range"
              min="100"
              max="4000"
              step="100"
              value={defaultMaxTokens}
              onChange={(e) => setDefaultMaxTokens(Number(e.target.value))}
            />
          </div>
          
          <div className="config-item">
            <label>Parallelism: {parallelism}</label>
            <input
              type="range"
              min="1"
              max="10"
              value={parallelism}
              onChange={(e) => setParallelism(Number(e.target.value))}
            />
          </div>
          
          <div className="config-item">
            <label>
              <input
                type="checkbox"
                checked={retryFailed}
                onChange={(e) => setRetryFailed(e.target.checked)}
              />
              Retry Failed Requests
            </label>
          </div>
          
          {retryFailed && (
            <div className="config-item">
              <label>Max Retries: {maxRetries}</label>
              <input
                type="range"
                min="1"
                max="5"
                value={maxRetries}
                onChange={(e) => setMaxRetries(Number(e.target.value))}
              />
            </div>
          )}
        </div>
      </div>

      <div className="prompts-section">
        <div className="prompts-header">
          <h3>Prompts ({prompts.length})</h3>
          <button onClick={addPrompt} className="add-prompt-button">
            <FilePlus size={16} />
            Add Prompt
          </button>
        </div>
        
        <div className="prompts-list">
          {prompts.map((prompt, index) => (
            <div key={index} className="prompt-item">
              <div className="prompt-header">
                <input
                  type="text"
                  value={prompt.id}
                  onChange={(e) => updatePrompt(index, 'id', e.target.value)}
                  placeholder="Prompt ID"
                  className="prompt-id"
                />
                <button onClick={() => removePrompt(index)} className="remove-button">
                  <XCircle size={16} />
                </button>
              </div>
              
              <textarea
                value={prompt.content}
                onChange={(e) => updatePrompt(index, 'content', e.target.value)}
                placeholder="Enter prompt content..."
                rows={3}
              />
              
              <details className="prompt-overrides">
                <summary>Override defaults</summary>
                <div className="override-grid">
                  <div className="override-item">
                    <label>Model</label>
                    <select 
                      value={prompt.model || ''} 
                      onChange={(e) => updatePrompt(index, 'model', e.target.value || null)}
                    >
                      <option value="">Use default</option>
                      {models.map(model => (
                        <option key={model.id} value={model.id}>{model.id}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div className="override-item">
                    <label>Temperature</label>
                    <input
                      type="number"
                      min="0"
                      max="2"
                      step="0.1"
                      value={prompt.temperature || ''}
                      onChange={(e) => updatePrompt(index, 'temperature', e.target.value ? Number(e.target.value) : null)}
                      placeholder="Default"
                    />
                  </div>
                  
                  <div className="override-item">
                    <label>Max Tokens</label>
                    <input
                      type="number"
                      min="100"
                      max="4000"
                      value={prompt.max_tokens || ''}
                      onChange={(e) => updatePrompt(index, 'max_tokens', e.target.value ? Number(e.target.value) : null)}
                      placeholder="Default"
                    />
                  </div>
                </div>
              </details>
            </div>
          ))}
        </div>
      </div>

      <button 
        onClick={runBatch} 
        disabled={loading || prompts.length === 0}
        className="run-button"
      >
        {loading ? (
          <>
            <RefreshCw size={16} className="spin" />
            Processing...
          </>
        ) : (
          <>
            <Play size={16} />
            Run Batch ({prompts.filter(p => p.content?.trim()).length} prompts)
          </>
        )}
      </button>

      {status && loading && (
        <div className="batch-progress">
          <h4>Processing Batch...</h4>
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${status.progress.percent}%` }}
            />
          </div>
          <div className="progress-stats">
            <span className="stat">
              <CheckCircle size={16} color="#4caf50" />
              Completed: {status.progress.completed}
            </span>
            <span className="stat">
              <XCircle size={16} color="#f44336" />
              Failed: {status.progress.failed}
            </span>
            <span className="stat">
              Total: {status.progress.total}
            </span>
          </div>
        </div>
      )}

      {results && (
        <div className="batch-results">
          <div className="results-header">
            <h3>Batch Results</h3>
            <button onClick={downloadResults} className="download-button">
              <Download size={16} />
              Download Results
            </button>
          </div>
          
          <div className="results-summary">
            <div className="summary-card">
              <h4>Summary</h4>
              <div className="summary-stats">
                <div className="stat-item">
                  <span className="stat-label">Status:</span>
                  <span className={`stat-value ${results.status}`}>{results.status}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Total Prompts:</span>
                  <span className="stat-value">{results.progress.total}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Successful:</span>
                  <span className="stat-value success">{results.progress.completed}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Failed:</span>
                  <span className="stat-value error">{results.progress.failed}</span>
                </div>
              </div>
            </div>
          </div>

          {results.results.length > 0 && (
            <div className="results-details">
              <h4>Successful Results</h4>
              <div className="results-list">
                {results.results.map((result, index) => (
                  <div key={index} className="result-item success">
                    <div className="result-header">
                      <span className="result-id">{result.prompt_id}</span>
                      <span className="result-model">{result.model}</span>
                      <span className="result-time">{result.duration_ms}ms</span>
                    </div>
                    <div className="result-content">
                      <details>
                        <summary>Response</summary>
                        <pre>{result.response}</pre>
                      </details>
                    </div>
                    {result.usage && (
                      <div className="result-usage">
                        <span>Tokens: {result.usage.total_tokens}</span>
                        <span>TPS: {result.usage.completion_tokens && result.duration_ms ? 
                          ((result.usage.completion_tokens / result.duration_ms) * 1000).toFixed(1) : 'N/A'}</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {results.errors.length > 0 && (
            <div className="results-details">
              <h4>Failed Results</h4>
              <div className="results-list">
                {results.errors.map((error, index) => (
                  <div key={index} className="result-item error">
                    <div className="result-header">
                      <span className="result-id">{error.prompt_id}</span>
                      <span className="result-attempts">Attempts: {error.attempts}</span>
                    </div>
                    <div className="result-error">
                      <AlertCircle size={16} />
                      {error.error}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default BatchProcessor;